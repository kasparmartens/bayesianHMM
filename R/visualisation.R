#' @export
visualise_gaussian_obs = function(hmm_obs){
  df = data.frame(t = 1:length(hmm_obs$x), x = hmm_obs$x, y = hmm_obs$y)
  p = ggplot(df, aes(t, y, col=x)) + geom_point() + 
    theme_bw() + xlab("") + theme(legend.position = "none") + 
    scale_color_brewer(palette="Accent")
  p_hist = ggplot(df, aes(y)) + geom_histogram() + 
    theme_bw() + xlab("") + coord_flip() 
  p_x = ggplot(df, aes(factor(1), y = t, fill=x)) + 
    geom_tile() + 
    coord_flip() + theme_bw() + xlab("t") + ylab("") + theme(legend.position = "none") + 
    scale_x_discrete(breaks=NULL) + 
    scale_fill_brewer(palette="Accent")
  
  mat <- matrix(list(ggplotGrob(p), ggplotGrob(p_x), ggplotGrob(p_hist), nullGrob()), nrow = 2)
  g <- gtable_matrix("demo", mat, unit(c(8, 2), "null"), unit(c(8, 1), "null"))
  grid.newpage()
  grid.draw(g)
}

#' @export
visualise_discrete_obs = function(hmm_obs){
  df = data.frame(t = 1:length(hmm_obs$x), x = hmm_obs$x, y = hmm_obs$y)
  p1 = ggplot(df, aes(factor(1), y = 1, fill=x)) + 
    geom_bar(stat="identity") + 
    coord_flip() + theme_bw() + xlab("") + ylab("") +  
    scale_x_discrete(breaks=NULL) + 
    scale_fill_brewer(palette="Accent") + ggtitle("Latent sequence x")
  p2 = ggplot(df, aes(factor(1), y = 1, fill=y)) + 
    geom_bar(stat="identity") + 
    coord_flip() + theme_bw() + xlab("") + ylab("") +  
    scale_x_discrete(breaks=NULL) + ggtitle("Observed sequence y")
  g = grid.arrange(p1, p2, ncol=1)
  return(g)
}

#' @export
visualise_traces_of_x = function(res_list, names_res, which_chain = 1, subsample = 2, subsequence = NULL){
  if(is.null(subsequence)) subsequence = 1:ncol(res_list[[1]]$trace_x[[1]])
  temp = lapply(1:length(res_list), function(i){
    tr = res_list[[i]]$trace_x[[which_chain]]
    tr_sub = tr[seq(1, nrow(tr), subsample), subsequence, drop=FALSE]
    rownames(tr_sub) = seq(1, nrow(tr), subsample)
    x.m = melt(tr_sub)
    x.m %>%
      mutate(value = factor(value), 
             type = names_res[i])
  })
  df = do.call("rbind", temp) %>%
    mutate(type = factor(type, levels = names_res))
  p = ggplot(df, aes(Var2, Var1)) + geom_tile(aes(fill=value)) + 
    theme_bw() + xlab("t") + ylab("Iteration") + facet_grid(. ~ type) + 
    scale_fill_brewer(palette="Accent")
  return(p)
}
