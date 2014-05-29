library(ggplot2);
library(reshape2);

fPath <- "C:/Users/Dillon/CWorkspace/SIMDStuff"
setwd(fPath);

filename <- "gemm_time_cmp_mmmul_1_mmmul_8.csv"

timeData <- read.csv(filename, header=T, sep=",");

timeData.m <- melt(timeData, id.var=1);

ggplot(timeData.m, aes(dim, value, colour = variable)) +geom_point();