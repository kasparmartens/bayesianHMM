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
visualise_trace_of_x <- function(x, which_chain = 1, subsample = 2, subsequence = NULL, all_chains = FALSE) UseMethod("visualise_trace_of_x")

#' @export
visualise_trace_of_x.ensembleHMM = function(res, which_chain = 1, subsample = 2, subsequence = NULL, all_chains = FALSE){
  if(is.null(subsequence)) subsequence = 1:ncol(res$trace_x[[1]])
  if(all_chains){
    temp = lapply(1:length(res$trace_x), function(i){
      tr = res$trace_x[[i]]
      tr_sub = tr[seq(1, nrow(tr), subsample), subsequence, drop=FALSE]
      rownames(tr_sub) = seq(1, nrow(tr), subsample)
      x.m = melt(tr_sub)
      x.m %>%
        mutate(value = factor(value), 
               type = paste("chain", i, sep="_"))
    })
    df = do.call("rbind", temp) %>%
      mutate(type = factor(type))
    p_base = ggplot(df, aes(Var2, Var1)) + 
      geom_tile(aes(fill=value)) + 
      facet_grid(. ~ type)
  }
  else{
    tr = res$trace_x[[which_chain]]
    tr_sub = tr[seq(1, nrow(tr), subsample), subsequence, drop=FALSE]
    rownames(tr_sub) = seq(1, nrow(tr), subsample)
    x.m = melt(tr_sub)
    df = x.m %>%
      mutate(value = factor(value))
    p_base = ggplot(df, aes(Var2, Var1)) + 
      geom_tile(aes(fill=value))
  }
  p = p_base + 
    theme_bw() + xlab("t") + ylab("Iteration") + 
    scale_fill_brewer(palette="Accent")
  return(p)
}

#' @export
visualise_trace_of_x.FHMM = function(res, true_x, which_chain = 1, subsample = 2, subsequence = NULL, all_chains = FALSE){
  if(is.null(subsequence)) subsequence = 1:ncol(res$trace_x[[1]])
  if(all_chains){
    temp = lapply(1:length(res$trace_X), function(i){
      tr = do.call("rbind", lapply(res$trace_X[[i]], function(x){
        colSums(abs(x - true_x))
      }))
      tr_sub = tr[seq(1, nrow(tr), subsample), subsequence, drop=FALSE]
      rownames(tr_sub) = seq(1, nrow(tr), subsample)
      x.m = melt(tr_sub)
      x.m %>%
        mutate(value = factor(value), 
               type = paste("chain", i, sep="_"))
    })
    df = do.call("rbind", temp) %>%
      mutate(type = factor(type))
    p_base = ggplot(df, aes(Var2, Var1)) + 
      geom_tile(aes(fill=value)) + 
      facet_grid(. ~ type)
  }
  else{
    # tr = res$trace_X[[which_chain]]
    tr = do.call("rbind", lapply(res$trace_X[[which_chain]], function(x){
      colSums(abs(x - true_x))
    }))
    tr_sub = tr[seq(1, nrow(tr), subsample), subsequence, drop=FALSE]
    rownames(tr_sub) = seq(1, nrow(tr), subsample)
    x.m = melt(tr_sub)
    df = x.m %>%
      mutate(value = factor(value))
    p_base = ggplot(df, aes(Var2, Var1)) + 
      geom_tile(aes(fill=value))
  }
  p = p_base + 
    theme_bw() + xlab("t") + ylab("Iteration") + 
    scale_fill_brewer(palette="Blues") +
    ggtitle("Hamming distance from the truth")
  return(p)
}

#' @export
visualise_trace_of_X = function(res, which_chain = 1){
  tr = res$trace_X[[which_chain]]
  K = nrow(tr[[1]])
  df = do.call("rbind", lapply(1:K, function(i){
    mat = do.call("rbind", lapply(tr, function(x)x[i, ]))
    mat.m = melt(mat)
    data.frame(mat.m, row = i)
  }))
  
  p = ggplot(df, aes(Var2, Var1, fill=factor(value))) + 
    geom_tile() + 
    facet_grid(. ~ row) + 
    theme_bw() + 
    scale_fill_manual(values = c("white", "grey10")) + 
    theme(legend.position = "none")
  return(p)
}

#' @export
visualise_traces_of_X = function(res_list, names, which_chain = 1){
  df = do.call("rbind", lapply(seq_along(res_list), function(j){
    res = res_list[[j]]
    tr = res$trace_X[[which_chain]]
    K = nrow(tr[[1]])
    df = do.call("rbind", lapply(1:K, function(i){
      mat = do.call("rbind", lapply(tr, function(x)x[i, ]))
      mat.m = melt(mat)
      data.frame(mat.m, row = i, type = names[j])
    }))
  }))
  
  p = ggplot(df, aes(Var2, Var1, fill=factor(value))) + 
    geom_tile() + 
    facet_grid(type ~ row) + 
    theme_bw() + 
    scale_fill_manual(values = c("white", "grey10")) + 
    theme(legend.position = "none")
  return(p)
}


#' @export
visualise_traces_of_x = function(res_list, names_res, which_chain = 1, subsample = 2, subsequence = NULL, all_chains = FALSE){
  if(is.null(subsequence)) subsequence = 1:ncol(res_list[[1]]$trace_x[[1]])
  if(all_chains){
    temp = lapply(1:length(res_list[[1]]$trace_x), function(i){
      tr = res_list[[1]]$trace_x[[i]]
      tr_sub = tr[seq(1, nrow(tr), subsample), subsequence, drop=FALSE]
      rownames(tr_sub) = seq(1, nrow(tr), subsample)
      x.m = melt(tr_sub)
      x.m %>%
        mutate(value = factor(value), 
               type = paste(names_res, "chain", i, sep="_"))
    })
    df = do.call("rbind", temp) %>%
      mutate(type = factor(type))
  }
  else{
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
  }
  p = ggplot(df, aes(Var2, Var1)) + geom_tile(aes(fill=value)) + 
    theme_bw() + xlab("t") + ylab("Iteration") + facet_grid(. ~ type) + 
    scale_fill_brewer(palette="Accent")
  return(p)
}

#' @export
plot_trace_of_mu = function(obj, which_chain = 1){
  mat = obj$trace_mu[[which_chain]]
  df = melt(mat)
  ggplot(df, aes(Var1, value)) + 
    geom_line() + 
    facet_wrap(~ Var2) + 
    theme_bw()
}

#' @export
plot_traces_of_mu = function(lst, names, which_chain = 1){
  df = do.call("rbind", lapply(seq_along(lst), function(i){
    obj = lst[[i]]
    mat = obj$trace_mu[[which_chain]]
    melt(mat) %>%
      mutate(type = names[i])
  }))
  
  ggplot(df, aes(Var1, value)) + 
    geom_line() + 
    facet_grid(type ~ Var2) + 
    theme_bw()
}

#' @export
plot_logposterior = function(obj, which_chain = 1){
  x = obj$log_posterior_cond[[which_chain]]
  df = data.frame(x = x, iter = obj$iter)
  ggplot(df, aes(iter, x)) + 
    geom_line() + theme_bw() + 
    ylab("log-posterior")
}

#' @export
plot_logposteriors = function(lst, names_res, which_chain = 1){
  df = do.call("rbind", lapply(seq_along(lst), function(i){
    obj = lst[[i]]
    x = obj$log_posterior_cond[[which_chain]]
    data.frame(x = x, iter = obj$iter, type=names_res[i])
  }))
  ggplot(df, aes(iter, x)) + 
    geom_line() + facet_wrap(~type)+
    theme_bw() + 
    ylab("log-posterior")
}

#' @export
visualise_crossovers = function(obj, names = NULL){
  if(is.list(obj)){
    df.m = do.call("rbind", lapply(seq_along(obj), function(i){
      res = obj[[i]]
      mat = visualise_crossovers_helper(res)
      df.m = melt(mat) %>%
        rename(Iteration = Var1, t = Var2) %>%
        mutate(type = names[i])
    })) 
    df.m = df.m %>%
      mutate(type = factor(type, levels = names))
    p0 = ggplot(df.m, aes(t, Iteration, fill=factor(value))) + 
      geom_tile() + 
      facet_wrap(~ type, nrow=1)
  } else{
    mat = visualise_crossovers_helper(obj)
    df.m = melt(mat) %>%
      rename(Iteration = Var1, t = Var2)
    p0 = ggplot(df.m, aes(t, Iteration, fill=factor(value))) + 
      geom_tile()
  }
  p = p0 + 
    theme_bw() +
    scale_fill_manual(values = c("white", "grey20", "#3182bd")) + 
    theme(legend.position = "none")
  return(p)
}

visualise_crossovers_helper = function(res){
  n = ncol(res$trace_x[[1]])
  mat = do.call("rbind", lapply(res$crossovers, function(x){
    out = rep(0, n)
    x[1:2] = sort(x[1:2])
    if(!is.null(x)){
      # if not flipped
      if(x[3] == 0){
        if(x[1] == x[2]){
          out = rep(0, n)
        } else{
          out[(x[1]+1):x[2]] = 1
        }
      } 
      # if flipped
      else{
        if(x[1] == x[2]){
          out = rep(2, n)
        }else{
          out[-c((x[1]+1):x[2])] = 2
        }
      }
    }
    return(out)
  }))
  return(mat)
}