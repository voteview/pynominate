#!/usr/bin/python
import sys
from pprint import pprint

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.stats import norm
from random import uniform

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


# -----
# Constraint functions
# -----
# These functions help constrain the space for optimizers
# that need constraints

def _smooth_min(a, b, k=100):
    """Need to smooth the min function for the constraint so that it has nice
    derivatives when the bill and SQ are the same distance from the origin."""
    # JBL: logaddexp avoids underflows...
    res = np.logaddexp(-k*a, -k*b)
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', r'divide by zero')
        return -res/k


def _smooth_min_grad(a, b, k=100):
    return np.exp(-k*np.array((a, b)) - np.logaddexp(-k*a, -k*b))


def circle_constraint_bp(bp):
    """Constrains bill and status quo midpoints to fall in the unit circle as
    well as one of the alternatives."""
    return [1.0 - np.sum(np.square(bp[0:2])),
            1.0 - _smooth_min(np.sum(np.square(np.array([bp[0]-bp[2], bp[1]-bp[3]]))),
                              np.sum(np.square(np.array([bp[0]+bp[2], bp[1]+bp[3]]))))]


def circle_constraint_bp_grad(bp):
    """Not vectorized because bill parameters are updated one vote at a time."""
    smin_p = _smooth_min_grad(a=np.sum(np.square(np.array([bp[0] - bp[2], bp[1] - bp[2]]))),
                              b=np.sum(np.square(np.array([bp[0] + bp[2], bp[1] + bp[3]]))))
    gloc = np.array(((bp[0]-bp[2], bp[1]-bp[3], -(bp[0]-bp[2]), -(bp[1]-bp[3])),
                    ((bp[0]+bp[2], bp[1]+bp[3],   bp[0]+bp[2],    bp[1]+bp[3]))))
    return -2*np.vstack((np.concatenate((bp[0:2], np.zeros(2))),
                         np.matmul(smin_p, gloc)))


def circle_constraint_idpt_fixed(idpts):
    """Constrains all ideals point (x) to fall inside the unit circle."""
    cnst = 1.0-np.sum(np.square(idpts))
    return cnst


def circle_constraint_idpt_fixed_grad(idpts):
    """Gradient of the unit circle constraint on the ideal points."""
    grad = -2.0*idpts
    return grad


def circle_constraint_idpt_spline(par):
    """
    Constraints ideal points to be within the unit circle
    par: a 1D array of all ideal points for one member (length = 2xN_sessions)
    """
    idpts = np.reshape(par, (-1, 2))
    cnst = 1.0-np.sum(np.square(idpts), 1)
    return cnst


def circle_constraint_idpt_spline_grad(par):
    # TODO: confirm constraint gradient and constraint are right shape
    grad = -2.0 * block_diag(*[par[i:i+2] for i in xrange(0, len(par), 2)])
    return grad


# ------
# NOMINATE likelihood functions
# ------
def pr_yea(bp, x, w, b):
    pr_y = norm.cdf(dwnominate_Uy(bp, x, w, b) - dwnominate_Un(bp, x, w, b))
    return pr_y


def dwnominate_Uy(bp, x, w, b):
    Uy = b * (np.exp(-((x[0] - bp[0] + bp[2])**2 +
                       w * w * (x[1] - bp[1] + bp[3])**2)))
    return Uy


def dwnominate_Uy_grad_x(Uy, bp, x, w):
    """Gradient of the utility associated with a yea vote given bill parameters and
    ideal points. Vectorized."""
    gDx = np.array([-2.0 * (x[0] - bp[0] + bp[2]),
                    -2.0 * w * w * (x[1] - bp[1] + bp[3])])
    g = Uy * gDx
    return g


def dwnominate_Uy_grad_bp(Un, bp, x, w):
    """Gradient of the utility associated with a yea vote given bill parameters and
    ideal points. Vectorized."""
    gDx = np.array([2.0 * (x[0] - bp[0] + bp[2]),
                    2.0 * w * w * (x[1] - bp[1] + bp[3]),
                    -2.0 * (x[0] - bp[0] + bp[2]),
                    -2.0 * w * w * (x[1] - bp[1] + bp[3])])
    g = Un * gDx
    return g


def dwnominate_Un(bp, x, w, b):
    Un = b * (np.exp(-((x[0] - bp[0] - bp[2])**2 +
                       w * w * (x[1] - bp[1] - bp[3])**2)))
    return Un


def dwnominate_Un_grad_x(Un, bp, x, w):
    """Gradient of the utility associated with a no vote given bill parameters and
    ideal points. Vectorized."""
    gDx = np.array((-2.0 * (x[0] - bp[0] - bp[2]),
                    -2.0 * w * w * (x[1] - bp[1] - bp[3])))
    g = Un * gDx
    return g


def dwnominate_Un_grad_bp(Un, bp, x, w):
    """Gradient of the utility associated with a no vote given bill parameters and
    ideal points. Vectorized."""
    gDx = np.array([2.0 * (x[0] - bp[0] - bp[2]),
                    2.0 * w * w * (x[1] - bp[1] - bp[3]),
                    2.0 * (x[0] - bp[0] - bp[2]),
                    2.0 * w * w * (x[1] - bp[1] - bp[3])])
    g = Un * gDx
    return g


def dwnominate_ll(bp, idpts, v, w, b, T):
    """Generic DW-NOMINATE log likelihood"""
    # When updating ideal points...
    if T is not None:
        design = (T[:, None] == np.array([range(np.amax(T)+1)])).astype(int)
        x = np.matmul(design, idpts).T
    # When updating votes...
    else:
        x = idpts
    ll = norm.logcdf(v*(dwnominate_Uy(bp, x, w, b) -
                        dwnominate_Un(bp, x, w, b)))
    return -np.sum(ll)


def dwnominate_ll_by_t(bp, idpts, v, w, b, T):
    """Generic DW-NOMINATE log likelihood"""
    if T is not None:
        design = (T[:, None] == np.array([range(np.amax(T)+1)])).astype(int)
        x = np.matmul(design, idpts).T
    ll = norm.logcdf(v*(dwnominate_Uy(bp, x, w, b) -
                        dwnominate_Un(bp, x, w, b)))
    ll = np.matmul(ll, design)
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', r'invalid')
        gmps = np.exp(ll / np.sum(design, axis=0))
    return zip(-1*ll, gmps)


def dwnominate_ll_bp(par, d, w, b):
    """Loglik to call when updating bill params"""
    x = d['ideal']
    v = d['votes']
    outofbounds = (par[0] * par[0] + par[1] * par[1]) > 1
    return dwnominate_ll(par, x, v, w, b, None) + (outofbounds and 1e300 or 0.0)


def dwnominate_ll_bp_grad(par, d, w, b):
    """Gradient of the logliklihood used when estimating the
    spline-smoothed ideal points."""
    x = d['ideal']
    v = d['votes']
    Uy = dwnominate_Uy(par, x, w, b)
    Un = dwnominate_Un(par, x, w, b)
    dd = v*(Uy-Un)
    return np.sum(-(v*norm.pdf(dd)/norm.cdf(dd)) * (
              dwnominate_Uy_grad_bp(Uy, par, x, w) -
              dwnominate_Un_grad_bp(Un, par, x, w)), axis=1)


#
# Functions for fixed/constant/single ideal point updates
#
def dwnominate_ll_idpt_fixed(idpts, d, w, b):
    """Loglik to be called when updating fixed ideal points."""
    bp = d['bp']
    v = d['votes']
    return dwnominate_ll(bp, idpts, v, w, b, None)


def dwnominate_ll_idpt_fixed_grad(idpts, d, w, b):
    """Gradient of the logliklihood used when estimating the
    spline-smoothed ideal points."""
    bp = d['bp']
    v = d['votes']
    Uy = dwnominate_Uy(bp, idpts, w, b)
    Un = dwnominate_Un(bp, idpts, w, b)
    dd = v*(Uy-Un)
    gradx = -(v*norm.pdf(dd)/norm.cdf(dd)) * (
        dwnominate_Uy_grad_x(Uy, bp, idpts, w) -
        dwnominate_Un_grad_x(Un, bp, idpts, w))
    return np.sum(gradx, axis=1)


def dwnominate_spline_penalty(idpts, lambdaval, n=1):
    """Spline penalty to apply to the likelihood when estimating ideal points."""
    max_t = np.size(idpts, 0)
    DD = np.diff(np.eye(max_t), n=n, axis=0)
    return lambdaval * np.sum(np.square(np.matmul(DD, idpts)))
 

def dwnominate_spline_penalty_grad(idpts, lambdaval, n=1):
    max_t = np.size(idpts, 0)
    DD = np.diff(np.eye(max_t), n=n, axis=0)
    return 2.0 * lambdaval * np.matmul(np.matmul(DD.T, DD), idpts)


def dwnominate_ll_idpt_spline(par, d, w, b, lambdaval, include_penalty=False, by_t=False):
    """Loglik to be called when updating ideal points.  Note that
    the penalty is applied here."""
    bp = d['bp']
    v = d['votes']
    t = d['t']
    idpts = np.reshape(par, (-1, 2))
    if by_t and include_penalty:
        raise Exception("Cannot set `by_t=True` and `include_penalty=True`")
    if by_t:
        return dwnominate_ll_by_t(bp, idpts, v, w, b, t)
    else:
        return dwnominate_ll(bp, idpts, v, w, b, t) + \
            (include_penalty and dwnominate_spline_penalty(idpts, lambdaval) or 0.0)

    
def dwnominate_ll_idpt_spline_grad(par, d, w, b, lambdaval):
    """Gradient of the logliklihood used when estimating the
    spline-smoothed ideal points.  `coefs` is the array of
    spline parameters."""
    bp = d['bp']
    v = d['votes']
    T = d['t']
    idpts = np.reshape(par, (-1, 2))
    design = (T[:, None] == np.array([range(np.amax(T)+1)])).astype(int)
    x = np.matmul(design, idpts).T
    Uy = dwnominate_Uy(bp, x, w, b)
    Un = dwnominate_Un(bp, x, w, b)
    dd = v*(Uy-Un)
    gradx = -(v*norm.pdf(dd)/norm.cdf(dd)) * (
              dwnominate_Uy_grad_x(Uy, bp, x, w) -
              dwnominate_Un_grad_x(Un, bp, x, w))
    gradx = np.matmul(design.T, gradx.T).flatten()
    return gradx + dwnominate_spline_penalty_grad(idpts, lambdaval).flatten()


def dwnominate_ll_idpt_star(args):
    """Wrapper to allow function to be called by map"""
    return dwnominate_ll_idpt(*args)


def dwnominate_ll_idpt(par, d, w, b, lambdaval=None, fixed=True, penalize_outofbounds=True):
    """Loglik to called when updating ideal points"""

    if fixed or lambdaval is None:
        ll_val = dwnominate_ll_idpt_fixed(par, d, w, b)
    else:
        ll_val = dwnominate_ll_idpt_spline(par, d, w, b, lambdaval)
    
    if penalize_outofbounds:
        outofbounds = (par[0] * par[0] + par[1] * par[1]) > 1
        return ll_val + (outofbounds and 1e300 or 0.0)
    else:
        return ll_val


def dwnominate_ll_wb(par, start, dat, pool):
    """Loglik to be called when updating ideal points"""
    args = zip(start, dat, [par[0]] * len(dat), [par[1]] * len(dat))
    ll = pool.map(dwnominate_ll_idpt_star, args, chunksize=100)
    return sum(ll) + (par[0] < 0 and 1e10 or 0) + (par[1] < 0 and 1e300 or 0)


def update_bp(d, w, b, par0=np.array([0, 0, 0, 0]), opt_method="SLSQP"):
    """update bill parameters for a single vote """
    
    # Check for valid start...
    dd = par0[0] * par0[0] + par0[1] * par0[1]
    if dd > 1:
        par0 = [p / (dd + 0.001) for p in par0]
    if opt_method == "SLSQP":
        opt_constraint = {"type": "ineq",
                          "fun": circle_constraint_bp,
                          "jac": circle_constraint_bp_grad}
        opt_jac = dwnominate_ll_bp_grad
    else:
        opt_constraint = None
        opt_jac = None
    try:
        llstart = dwnominate_ll_bp(par0, d, w, b)
        ares = minimize(dwnominate_ll_bp,
                        par0,
                        jac=opt_jac,
                        method=opt_method,
                        options=OPTIONS,
                        constraints=opt_constraint,
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
            u'status': ares['status'],
            u'bp': res}


def update_bp_star(args):
    """wrapper to call from map"""
    return update_bp(*args)


def update_idpt(d, w, b, par0, lambdaval=None, fixed=True, always_spline=False):
    """Update ideal points.  Use splines as required"""
    if fixed or lambdaval is None:
        par0 = np.array(par0[0:2])  # Only need two elements in starts.
        return update_idpt_fixed(d, w, b, par0)
    return update_idpt_spline(d, w, b, par0, lambdaval)


def update_idpt_spline(d, w, b, par0, lambdaval):
    """Update ideal points for a single member"""
    par0 = np.array(par0).flatten()
#    print(par0)
#    print(d)
#    print(w)
#    print(b)
#    print(lambdaval)
    llstart = dwnominate_ll_idpt_spline(par0, d, w, b, lambdaval=0)
    resd = minimize(
        fun=dwnominate_ll_idpt_spline,
        x0=par0,
        jac=dwnominate_ll_idpt_spline_grad,
        method="SLSQP",
        options={'maxiter': 1000},
        constraints={
            "type": "ineq",
            "fun": circle_constraint_idpt_spline,
            "jac": circle_constraint_idpt_spline_grad
        },
        args=(d, w, b, lambdaval)
    )
    res = resd['x']
    llend = dwnominate_ll_idpt_spline(res, d, w, b, lambdaval=0)
    ll_gmp = dwnominate_ll_idpt_spline(res, d, w, b, lambdaval=0, include_penalty=False, by_t=True)
    if np.exp(-llend/len(d['votes'])) < 0.5:
        print "WARNING: Bad ideal point fit encountered"
    return {u'llstart': llstart,
            u'llend': llend,
            u'status': resd['status'],
            u'message': resd['message'],
            u'll_gmp': ll_gmp,
            u'x': np.reshape(res, (-1, 2))}


def _random_start():
    """Draw random start values for ideal point. Uniformly distributed over
    a 'doughnut' inside the unit circle"""
    theta = uniform(0, 2 * np.pi)
    return uniform(0.2, 0.8) * np.array([np.cos(theta), np.sin(theta)])


def update_idpt_fixed(d, w, b, par0, attempt=0):
    """Update ideal points for a single member with fixed ideal point"""
    par0 = np.array(par0).flatten()
    llstart = dwnominate_ll_idpt_fixed(par0, d, w, b)
    resd = minimize(
        fun=dwnominate_ll_idpt_fixed,
        x0=par0,
        jac=dwnominate_ll_idpt_fixed_grad,
        method="SLSQP",
        options=OPTIONS,
        constraints={
            "type": "ineq",
            "fun": circle_constraint_idpt_fixed,
            "jac": circle_constraint_idpt_fixed_grad
        },
        args=(d, w, b)
    )
    res = resd['x']
    llend = dwnominate_ll_idpt_fixed(res, d, w, b)
    gmp = np.exp(-llend / len(d['votes']))
    if gmp < 0.5 and attempt < 10:
        attempt += 1
        print "Bad ideal point fit found! Trying again",
        print "(GMP = %4.2f)" % gmp
        return update_idpt_fixed(d, w, b, _random_start(), attempt)
    return {u'llstart': llstart,
            u'llend': llend,
            u'status': resd['status'],
            u'll_gmp': [[llend, gmp]],
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


def update_nominate(
        payload,
        update=['bp', 'idpt', 'bw'],
        lambdaval=None,
        maxiter=20,
        cores=int(cpu_count() - 1),
        xtol=1e-4,
        add_meta=['members', 'rollcalls']
):
    import time

    # OPTIONS['xtol'] = xtol
    # OPTIONSWB['xtol'] = xtol

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
    #mymap = map

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
            dat = [
                {
                    'votes': np.array(tuple(xx[0] for xx in payload['votes'][i]['votes'])),
                    'ideal': np.transpose(np.array([
                        payload['idpt'][str(xx[1])][str(int(payload['votes'][i]['id'][2:5]))]
                        if isinstance(payload['idpt'][str(xx[1])], dict)
                        else payload['idpt'][str(xx[1])]
                        for xx in payload['votes'][i]['votes']
                    ]))
                }
                for i in range(len(payload['votes'])) if payload['votes'][i]['update']
            ]
            start = [
                payload['bp'][v['id']]
                for v in payload['votes'] if payload['votes'][i]['update']
            ]
            print "(%03i) Rollcall update data marshal took %2.2f seconds (%i votes)..." % (iter + 1, time.time() - starttime, len(start))
            res_bp = mymap(
                update_bp_star,
                zip(
                    dat,
                    [w] * len(dat),
                    [b] * len(dat),
                    start
                )
            )
            for i, v in enumerate(payload['votes']):
                if v['update']:
                    payload['bp'][v['id']] = res_bp[i]['bp']
            print "\t\t" + str(res_bp[0]['bp'])
            print "(%03i) Rollcall update took %2.2f seconds (%i votes)..." % (iter + 1, time.time() - starttime, len(start))

        # Update member
        if 'idpt' in update or 'bw' in update:
            starttime = time.time()
            dat = [
                {
                    'votes': np.array(tuple(xx[0] for xx in payload['memberwise'][i]['votes'])),
                    'bp': np.transpose(np.array([
                        payload['bp'][xx[1]]
                        for xx in payload['memberwise'][i]['votes']
                    ])),
                    't': np.array(tuple(int(xx[2]) for xx in payload['memberwise'][i]['votes']))
                }
                for i in range(len(payload['memberwise'])) if payload['memberwise'][i]['update']
            ]
            # build idpt matrix
            start = [
                payload['idpt'][v['icpsr']]['idpts']
                for v in payload['memberwise'] if v['update']
            ]
            print "(%03i) Member/BW update data marshal took %2.2f seconds (%i members)..." % (iter + 1, time.time() - starttime, len(start))

        if 'idpt' in update:
            res_idpt = mymap(
                update_idpt_star,
                zip(
                    dat,
                    [w] * len(dat),
                    [b] * len(dat),
                    start,
                    [lambdaval] * len(dat),
                    [False] * len(dat)
                )
            )
            for i, v in enumerate(payload['memberwise']):
                if v['update']:
                    payload['idpt'][v['icpsr']]['idpts'] = res_idpt[i]['x']

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
                ret_bp[v['id']] = {
                    'bp': list(res_bp[i]['bp']),
                    'log_likelihood': -res_bp[i]['llend'],
                    'gmp': np.exp(-res_bp[i]['llend'] / len(v['votes'])),
                    'status': res_bp[i]['status']
                }
        ret['bp'] = ret_bp
    if 'idpt' in update:
        ret_idpt = {}
        for i, v in enumerate(payload['memberwise']):
            if v['update']:
                ret_idpt[v['icpsr']] = {
                    'idpts': res_idpt[i]['x'],
                    'cong_range': tuple(
                        payload['idpt'][v['icpsr']][t + '_cong']
                        for t in ['min', 'max']
                    ),
                    'meta': {
                        'all': {
                            'log_likelihood': -res_idpt[i]['llend'],
                            'gmp': np.exp(-res_idpt[i]['llend'] / len(v['votes']))
                        },
                        'by_t': res_idpt[i]['ll_gmp'],
                        'status': res_idpt[i]['status'],
                        'message': res_idpt[i]['message']
                    },
                }
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

    pool.close()
    pool.join()
    
    return ret


if __name__ == '__main__':
    print('Running pynominate')

    if True:
        test_payload = {
            'votes': [
                {
                    'votes': [[-1, "1"], [1, "2"], [1, "3"]],
                    'id': "RS09901",
                    'update': 1
                },
                {
                    'votes': [[-1, "1"], [1, "2"], [-1, "3"]],
                    'id': "RS09902",
                    'update': 1
                },
                {
                    'votes': [[-1, "1"], [1, "2"], [1, "3"]],
                    'id': "RS10001",
                    'update': 1
                },
                {
                    'votes': [[-1, "1"], [1, "2"], [1, "3"]],
                    'id': "RS10002",
                    'update': 1
                }
            ],
            'bp': {
                'RS09901': [0, 0, 0, 0],
                'RS09902': [0, 0, 0, 0],
                'RS10001': [0, 0, 0, 0],
                'RS10002': [0, 0, 0, 0],
            },
            'memberwise': [
                {
                    'icpsr': "1",
                    'votes': [
                        [-1, "RS09901", 0],
                        [-1, "RS09902", 0],
                        [-1, "RS10001", 1],
                        [-1, "RS10002", 1],
                    ],
                    'update': 1
                },
                {
                    'icpsr': "2",
                    'votes': [
                        [1, "RS09901", 0],
                        [1, "RS09902", 0],
                        [1, "RS10001", 1],
                        [1, "RS10002", 1],
                    ],
                    'update': 1
                },
                {
                    'icpsr': "3",
                    'votes': [
                        [1, "RS09901", 0],
                        [-1, "RS09902", 0],
                        [1, "RS10001", 1],
                        [1, "RS10002", 1],
                    ],
                    'update': 1
                },
            ],
            'idpt': {
                "1": {
                    "99": [-0.5, 0],
                    "100": [-0.5, 0]
                },
                "2": {
                    "99": [0, 0],
                    "100": [0.5, 0]
                },
                "3": {
                    "99": [0, 0],
                    "100": [0, 0],
                }
            }
        }
        
        pprint(
            update_nominate(
                test_payload,
                update=['bp', 'idpt'],
                lambdaval=1,
                maxiter=2,
                add_meta=[]
            )
        )
        
