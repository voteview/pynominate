#!/usr/bin/python
import sys
import json
import numpy as np
from scipy.optimize import minimize
from scipy.stats import logistic, norm
import urllib
from multiprocessing import cpu_count
from multiprocessing import Pool


def cons(x):
    return 1 - (x[0] * x[0] + x[1] * x[1])

METHOD = 'Nelder-Mead'
OPTIONS = {'xtol': 1e-4, 'disp': False}
CONSTRAINTS = None

METHODWB = METHOD
OPTIONSWB = OPTIONS
CONSTRAINTSWB = None


def pr_yea(bp, x, w, b):
    pr_y = norm.cdf(dwnominate_Uy(bp, x, w, b) - dwnominate_Un(bp, x, w, b))
    return pr_y


def dwnominate_Uy(bp, x, w, b):
    Uy = b * (np.exp(-((x[0] - bp[0] + bp[2])**2 +
                       w * w * (x[1] - bp[1] + bp[3])**2)))
    return Uy


def dwnominate_Un(bp, x, w, b):
    Un = b * (np.exp(-((x[0] - bp[0] - bp[2])**2 +
                       w * w * (x[1] - bp[1] - bp[3])**2)))
    return Un


def dwnominate_ll(bp, x, v, w, b):
    """Generic DW-NOMINATE log likelihood"""
    Uy = dwnominate_Uy(bp, x, w, b)
    Un = dwnominate_Un(bp, x, w, b)
    ll = norm.logcdf(v * (Uy - Un))
    return -np.sum(ll)


def dwnominate_ll_bp(par, d, w, b):
    """Loglik to call when updating bill params"""
    x = d['ideal']
    v = d['votes']
    outofbounds = (par[0] * par[0] + par[1] * par[1]) > 1
    return dwnominate_ll(par, x, v, w, b) + (outofbounds and 1e300 or 0.0)


def dwnominate_ll_idpt(par, d, w, b):
    """Loglik to called when updating ideal points"""
    bp = d['bp']
    v = d['votes']
    outofbounds = (par[0] * par[0] + par[1] * par[1]) > 1
    return dwnominate_ll(bp, par, v, w, b) + (outofbounds and 1e300 or 0.0)


def dwnominate_ll_idpt_star(args):
    """Wrapper to allow function to be called by map"""
    return dwnominate_ll_idpt(*args)


def dwnominate_ll_wb(par, start, dat, pool):
    """Loglik to be called when updating ideal points"""
    args = zip(start, dat, [par[0]] * len(dat), [par[1]] * len(dat))
    ll = pool.map(dwnominate_ll_idpt_star, args, chunksize=100)
    return sum(ll) + (par[0] < 0 and 1e10 or 0) + (par[1] < 0 and 1e300 or 0)


def update_bp(d, w, b, par0=np.array([0, 0, 0, 0])):
    """update bill parameters for a single vote """

    # Check for valid start...
    dd = par0[0] * par0[0] + par0[1] * par0[1]
    if dd > 1:
        par0 = [p / (dd + 0.001) for p in par0]

    try:
        llstart = dwnominate_ll_bp(par0, d, w, b)
        ares = minimize(dwnominate_ll_bp,
                        par0,
                        method=METHOD,
                        options=OPTIONS,
                        constraints=CONSTRAINTS,
                        args=(d, w, b))
        res = ares['x']
        llend = dwnominate_ll_bp(res, d, w, b)

    except Exception, e:
        print "\t\tError: %s" % e
        print "\t\tProblem in: %s" % d
        print "\t\tReturning start values and moving on..."
        llstart = 0
        llend = 0
        res = par0

    return {u'llstart': llstart,
            u'llend': llend,
            u'bp': res}


def update_bp_star(args):
    """wrapper to call from map"""
    return update_bp(*args)


def update_idpt(d, w, b, par0=np.array([0, 0, 0, 0])):
    """Update ideal points for a single member"""

    # Check for valid start...
    dd = par0[0] * par0[0] + par0[1] * par0[1]
    if dd > 1:
        par0 = [p / (dd + 0.001) for p in par0]

    llstart = dwnominate_ll_idpt(par0, d, w, b)
    res = minimize(dwnominate_ll_idpt,
                   par0,
                   method=METHOD,
                   options=OPTIONS,
                   constraints=CONSTRAINTS,
                   args=(d, w, b))['x']
    llend = dwnominate_ll_idpt(res, d, w, b)
    return {u'llstart': llstart,
            u'llend': llend,
            u'x': res}


def update_idpt_star(args):
    """Wrapper to allow call from map"""
    return update_idpt(*args)


def update_wb(d, start, w0, b0, pool):
    """Update w and b parameters"""
    par0 = (w0, b0)
    res = minimize(dwnominate_ll_wb,
                   par0,
                   method=METHODWB,
                   options=OPTIONSWB,
                   constraints=CONSTRAINTSWB,
                   args=(start, d, pool))
    return {u'llend': res['fun'],
            u'w': res['x'][0],
            u'b': res['x'][1]}


def geo_mean_probability(ll, number_of_votes):
    return np.exp(ll / number_of_votes)


def add_rollcall_meta(payload, ret):
    for v in payload['votes']:
        if 'update' in v and v['update']:
            ret['bp'][v['id']]['geo_mean_probability'] = geo_mean_probability(
                ret['bp'][v['id']]['log_likelihood'], len(v['votes']))
    return ret


def add_member_meta(payload, ret, by_congress=True):
    if 'w' in ret:
        w = ret['w']
        b = ret['b']
    else:
        w = payload['bw']['w']
        b = payload['bw']['b']

    for m in payload['memberwise']:
        if 'update' in m and m['update']:
            idpt = ret['idpt'][m['icpsr']]['idpt']
            if by_congress:
                votes = {}
                for v in m['votes']:
                    chamber_congress_id = v[1][1:5]
                    if chamber_congress_id in votes:
                        votes[chamber_congress_id]['bp'].append(
                            payload['bp'][str(v[1])])
                        votes[chamber_congress_id]['votes'].append(v[0])
                    else:
                        votes[chamber_congress_id] = {'bp': [payload['bp'][str(v[1])]],
                                                      'votes': [v[0]]}

                all_dict = {'log_likelihood': 0.0,
                            'number_of_votes': 0,
                            'number_of_errors': 0}

                for chamber_congress, dat in votes.iteritems():
                    congress = int(chamber_congress[1:])
                    chamber = 'Senate' if chamber_congress[
                        0] == 'S' else 'House'
                    dat['bp'] = np.transpose(np.array(dat['bp']))
                    dat['votes'] = np.array(tuple(dat['votes']))
                    ll = -dwnominate_ll_idpt(idpt, dat, w, b)

                    meta_dict = {'log_likelihood': ll,
                                 'geo_mean_probability': geo_mean_probability(ll, len(dat['votes'])),
                                 'number_of_votes': len(dat['votes']),
                                 'number_of_errors': sum([1 if ((0.5 - pr_yea(dat['bp'][:, i], idpt, w, b)) * v) > 0 else 0 for i, v in enumerate(dat['votes'])])}

                    if congress in ret['idpt'][m['icpsr']]['meta']:
                        ret['idpt'][m['icpsr']]['meta'][
                            congress][chamber] = meta_dict
                    else:
                        ret['idpt'][m['icpsr']]['meta'][
                            congress] = {chamber: meta_dict}

                    all_dict['log_likelihood'] += meta_dict['log_likelihood']
                    all_dict['number_of_votes'] += meta_dict['number_of_votes']
                    all_dict['number_of_errors'] += meta_dict['number_of_errors']
            else:
                dat = {'votes': np.array(tuple(xx[0] for xx in m['votes'])),
                       'bp': np.transpose(np.array([payload['bp'][str(xx[1])]
                                                    for xx in m['votes']]))}

                ret['idpt'][m['icpsr']]['meta']['all']['number_of_errors'] = sum([1 if (
                    (0.5 - pr_yea(dat['bp'][:, i], idpt, w, b)) * v) > 0 else 0 for i, v in enumerate(dat['votes'])])

            ret['idpt'][m['icpsr']]['meta']['all']['geo_mean_probability'] = geo_mean_probability(
                ret['idpt'][m['icpsr']]['meta']['all']['log_likelihood'], len(m['votes']))
            ret['idpt'][m['icpsr']]['meta']['all'][
                'number_of_votes'] = len(m['votes'])

    return ret


def update_nominate(payload, maxiter=20, cores=int(cpu_count() - 1), update=['bp', 'idpt', 'bw'], xtol=1e-4, add_meta=['members', 'rollcalls']):
    import time

    OPTIONS['xtol'] = xtol
    OPTIONSWB['xtol'] = xtol

    if 'bp' in update and ('votes' not in payload or not payload['votes']):
        print "Payload missing 'votes'! Cannot update bill parameters, quitting."
        sys.exit(1)
    if 'idpt' in update and ('memberwise' not in payload or not payload['memberwise']):
        print "Payload missing 'memberwise'! Cannot update ideal points, quitting."
        sys.exit(1)

    print "(000) Running DW-NOMINATE on %i cores..." % cores
    firststarttime = time.time()

    if 'votes' in payload:
        nchoices = sum([len([xx for xx in x['votes'] if xx != 0])
                        for x in payload['votes']])
        print "(000) %i total vote choices observed..." % nchoices

    # Run dwnominate...
    pool = Pool(cores)
    mymap = pool.map  # allow switching to in/out parallel processing for debugging

    if 'bw' in payload:
        b = payload['bw']['b']
        w = payload['bw']['w']
    else:
        b = 8.8633
        w = 0.4619
    iter = 0
    while iter < maxiter:
        # Update roll call parameters...
        if 'bp' in update:
            starttime = time.time()
            dat = [{'votes': np.array(tuple(xx[0] for xx in payload['votes'][i]['votes'])),
                    'ideal':np.transpose(np.array([payload['idpt'][xx[1]]
                                                   for xx in payload['votes'][i]['votes']]))}
                   for i in range(len(payload['votes'])) if payload['votes'][i]['update']]
            start = [payload['bp'][v['id']]
                     for v in payload['votes'] if payload['votes'][i]['update']]
            print "(%03i) Rollcall update data marshal took %2.2f seconds (%i votes)..." % (iter + 1, time.time() - starttime, len(start))
            res_bp = mymap(update_bp_star, zip(
                dat, [w] * len(dat), [b] * len(dat), start))
            for i, v in enumerate(payload['votes']):
                if v['update']:
                    payload['bp'][v['id']] = res_bp[i]['bp']
            print "\t\t" + str(res_bp[0]['bp'])
            print "(%03i) Rollcall update took %2.2f seconds (%i votes)..." % (iter + 1, time.time() - starttime, len(start))

        # Update member
        if 'idpt' in update or 'bw' in update:
            starttime = time.time()
            dat = [{'votes': np.array(tuple(xx[0] for xx in payload['memberwise'][i]['votes'])),
                    'bp':np.transpose(np.array([payload['bp'][str(xx[1])]
                                                for xx in payload['memberwise'][i]['votes']]))}
                   for i in range(len(payload['memberwise'])) if payload['memberwise'][i]['update']]
            start = [payload['idpt'][v['icpsr']]
                     for v in payload['memberwise'] if v['update']]
            print "(%03i) Member/BW update data marshal took %2.2f seconds (%i members)..." % (iter + 1, time.time() - starttime, len(start))

        if 'idpt' in update:
            res_idpt = mymap(update_idpt_star, zip(
                dat, [w] * len(dat), [b] * len(dat), start))
            for i, v in enumerate(payload['memberwise']):
                if v['update']:
                    payload['idpt'][v['icpsr']] = res_idpt[i]['x']
            print "(%03i) Member update took %2.2f seconds (%i members)..." % (iter + 1, time.time() - starttime, len(start))
            print "\t\t Ideal Point[0] = " + str(res_idpt[0]['x'])

        # Update b and w
        if 'bw' in update:
            starttime = time.time()
            start = [payload['idpt'][v['icpsr']]
                     for v in payload['memberwise']]
            print "(%03i) Weight and Beta update data marshal took %2.2f seconds (%i members)..." % (iter + 1, time.time() - starttime, len(start))
            res_wb = update_wb(dat, start, w, b, pool)
            w, b = res_wb['w'], res_wb['b']
            print "(%03i) Weight and Beta  update took %2.2f seconds..." % (iter + 1, time.time() - starttime)
            print "\t\t w = %7.4f, b = %7.4f" % (w, b)
            print "(%03i) Iteration Loglik: -%9.4f across %i choices (GMP=%6.4f)\n" % (iter + 1, res_wb['llend'], nchoices, np.exp(-res_wb['llend'] / nchoices))
        iter += 1

    print "(XXX) Total update time elapsed %5.2f minutes." % ((time.time() - firststarttime) / 60)

    ret = {'control': {'iterations': iter,
                       'time': round((time.time() - firststarttime) / 60, 3),
                       'method': METHOD,
                       'options': OPTIONS,
                       'cores': cores}}
    if 'bp' in update:
        ret_bp = {}
        for i, v in enumerate(payload['votes']):
            if v['update']:
                ret_bp[v['id']] = {'bp': list(
                    res_bp[i]['bp']), 'log_likelihood': -res_bp[i]['llend']}
        ret['bp'] = ret_bp
    if 'idpt' in update:
        ret_idpt = {}
        for i, v in enumerate(payload['memberwise']):
            if v['update']:
                ret_idpt[v['icpsr']] = {'idpt': list(res_idpt[i]['x']), 'meta': {
                    'all': {'log_likelihood': -res_idpt[i]['llend']}}}
        ret['idpt'] = ret_idpt
    if 'bw' in update:
        ret['b'] = b
        ret['w'] = w

    if 'bp' in update and 'idpt' in update and 'bw' in update:
        ret['log_likelihood'] = round(-res_wb['llend'], 4)
        ret['GMP'] = np.exp(-res_wb['llend'] / nchoices)

    if 'idpt' in update and 'members' in add_meta:
        print "Adding member meta (GMP, number_of_votes, etc.)..."
        ret = add_member_meta(payload, ret)
    if 'bp' in update and 'rollcalls' in add_meta:
        print "Adding gmp to rollcalls..."
        ret = add_rollcall_meta(payload, ret)

    return ret


if __name__ == '__main__':
    print('Running pynominate')
