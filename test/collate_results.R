setwd("~/Dropbox/voteview/pynominate/")

library(jsonlite)
library(dplyr)

ses <- fromJSON("test/data/ses.json")
bres <- fromJSON("test/data/bs.json")
lens <- fromJSON("test/data/len.json")
ests <- fromJSON("test/idpt_payload.json")

boots <- do.call("rbind", lapply(names(bres), FUN = function(icpsr) {
  boot_ests <- bres[[icpsr]]
  colnames(boot_ests) <- c("boot_dim1", "boot_dim2")
  se <- matrix(ses[[icpsr]], nrow = 1)
  colnames(se) <- c("se_dim1", "se_dim2")
  est <- matrix(ests[['idpt']][[icpsr]], nrow = 1)
  colnames(est) <- c("dim1", "dim2")
  data.frame(
    icpsr = icpsr,
    bootstrap_it = seq_len(nrow(boot_ests)),
    est,
    se,
    boot_ests,
    nvotes = lens[[icpsr]],
    stringsAsFactors = FALSE
  )
}))

hsall <- read.csv("~/Downloads/HSall_members.csv", stringsAsFactors = FALSE)

boot_m <- boots %>%
  group_by(icpsr) %>%
  mutate(cor_bs = cor(boot_dim1, boot_dim2)) %>%
  summarize_all(mean) %>%
  mutate(dim1_diff = dim1 - boot_dim1,
         dim2_diff = dim2 - boot_dim2,
         abs_dim1_diff = abs(dim1 - boot_dim1),
         abs_dim2_diff = abs(dim2 - boot_dim2)) %>%
  merge(., hsall %>% group_by(icpsr) %>% filter(congress == max(congress)) %>% summarize(last_congress = mean(congress)), all.x = T)

write.csv(boot_m, file = "boot.csv", row.names = F)
