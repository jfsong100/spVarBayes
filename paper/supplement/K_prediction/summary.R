library(tidyr)
library(dplyr)
library(ggplot2)
library(readr)
library(stringr)
library(tidyverse)
library(gridExtra)

######################################
# Output path
######################################
base_dir  = file.path(getwd())
output.path  = file.path(base_dir,"figures")
if(!dir.exists(output.path)){
  dir.create(output.path)
}
##################################################
# Input spVB and spNNGP results from R_results
##################################################
file_list= list.files(file.path(base_dir,  "R_results"))
VI_pred_list = file_list[grepl("VI_NNGP_pred_list", file_list, fixed = TRUE)]
VI_pred_vector_list = file_list[grepl("VI_NNGP_pred_vector_t", file_list, fixed = TRUE)]

output_list_t = matrix(NA,ncol=42)
output_list_total = matrix(NA,ncol=16)


colnames(output_list_total) = c("t","n",
                                "y_test",
                                "NNGP_m3_y_test","NNGP_joint_m3_y_test",
                                "MFA_y_test", "MFA_LR_y_test",
                                "spNNGP_y_test",
                                "w_test",
                                "NNGP_m3_w_test","NNGP_joint_m3_w_test",
                                "MFA_w_test", "MFA_LR_w_test",
                                "spNNGP_w_test",
                                "index","n_train")

NNGPVI_path = file.path(base_dir,  "R_results")
setwd(NNGPVI_path)

for (i in 1:length(VI_pred_vector_list)) {

    filename = VI_pred_vector_list[i]
    n = str_match(filename, "_n(\\d+)")[,2] %>% as.integer()
    output_list_t_sub = unlist(read.csv(VI_pred_vector_list[i]))

    output_list_t = rbind(output_list_t,t(output_list_t_sub))
    sub_data_NNGP = read.csv(VI_pred_list[i])
    sub_data_NNGP$n_train = n
    output_list_total = rbind(output_list_total,sub_data_NNGP)
  

}

##################################################
# Input DKLGP_default and DKLGP results from DKLGP_results
##################################################
file_list_DKL= list.files(file.path(base_dir,  "DKLGP_results"))

VI_pred_list_DKL = c(file_list_DKL[grepl("output_pred_VIVA_n", file_list_DKL, fixed = TRUE)])
pred_list_DKL = file_list_DKL[
  grepl("^pred_VIVA_n", file_list_DKL) & 
    !grepl("^output_pred_VIVA_n", file_list_DKL)
]

file_list_DKL_default= list.files(file.path(base_dir,  "DKLGP_results"))
VI_pred_list_DKL_default = file_list_DKL_default[grepl("output_pred_VIVA_default", file_list_DKL_default, fixed = TRUE)]
pred_list_DKL_default = file_list_DKL_default[
  grepl("^pred_VIVA_default", file_list_DKL_default) & 
    !grepl("^output_pred_VIVA_default", file_list_DKL_default)
]


output_list_t_DKL = matrix(NA,ncol=8)
output_list_total_DKL = matrix(NA,ncol=8)

colnames(output_list_total_DKL) = c("t","n","index","w_pred","w_var","y_pred","y_var","n_train")

colnames(output_list_t_DKL) = c("seed","n","w_test_coverage","w_test_is_mean","w_test_crps_mean",
                                "y_test_coverage","y_test_is_mean","y_test_crps_mean")

setwd(file.path(base_dir,  "DKLGP_results"))
for (i in 1:length(pred_list_DKL)) {

    filename = pred_list_DKL[i]
    n = str_match(filename, "_n(\\d+)")[,2] %>% as.integer()
    
    output_list_t_sub = read.csv(pred_list_DKL[i])
    output_list_t_sub[2] = n
    output_list_t_DKL = rbind(output_list_t_DKL,output_list_t_sub)

    sub_data = read.csv(VI_pred_list_DKL[i])
    sub_data$n_train = n
    output_list_total_DKL = rbind(output_list_total_DKL,sub_data)
  

}

output_list_t_DKL_default = matrix(NA,ncol=8)
output_list_total_DKL_default = matrix(NA,ncol=8)

colnames(output_list_total_DKL_default) = c("t","n","index","w_pred","w_var","y_pred","y_var","n_train")

colnames(output_list_t_DKL_default) = c("seed","n","w_test_coverage","w_test_is_mean","w_test_crps_mean",
                                "y_test_coverage","y_test_is_mean","y_test_crps_mean")


options(warn = 1)

for (i in 1:length(pred_list_DKL_default)) {

    filename = pred_list_DKL_default[i]
    n = str_match(filename, "_n(\\d+)")[,2] %>% as.integer()
    
    output_list_t_sub = read.csv(pred_list_DKL_default[i])
    output_list_t_sub[2] = n
    
    output_list_t_DKL_default = rbind(output_list_t_DKL_default,output_list_t_sub)
    
    sub_data = read.csv(VI_pred_list_DKL_default[i])
    sub_data$n_train = n
    output_list_total_DKL_default = rbind(output_list_total_DKL_default,sub_data)
    
  
}

##################################################
# Input VNNGP results from VNNGP_results
##################################################
file_list_VNNGP= list.files(file.path(base_dir,  "VNNGP_results"))
VI_pred_list_VNNGP = file_list_VNNGP[grepl("output_pred_VNNGP", file_list_VNNGP, fixed = TRUE)]
pred_list_VNNGP = file_list_VNNGP[
  grepl("^pred_VNNGP", file_list_VNNGP) & 
    !grepl("^output_pred_VNNGP", file_list_VNNGP)
]

output_list_t_VNNGP = matrix(NA,ncol=8)
output_list_total_VNNGP = matrix(NA,ncol=8)

colnames(output_list_total_VNNGP) = c("t","n","index","w_pred","w_var","y_pred","y_var","n_train")

colnames(output_list_t_VNNGP) = c("seed","n","w_test_coverage","w_test_is_mean","w_test_crps_mean",
                                        "y_test_coverage","y_test_is_mean","y_test_crps_mean")

options(warn = 1)
setwd(file.path(base_dir,  "VNNGP_results"))

for (i in 1:length(pred_list_VNNGP)) {

  VNNGP_data = read_delim(pred_list_VNNGP[i])
  
  filename = pred_list_VNNGP[i]
  t = str_match(filename, "_seed(\\d+)")[,2] %>% as.integer()
  n = str_match(filename, "_n(\\d+)")[,2] %>% as.integer()
  VNNGP_data$n = n
  output_list_t_VNNGP = rbind(output_list_t_VNNGP,VNNGP_data)
  sub_data = read_delim(paste0("output_pred_VNNGP","_n",n,"_d2_seed",VNNGP_data[1],".txt"))
  sub_data$n_train = n
  output_list_total_VNNGP = rbind(output_list_total_VNNGP,sub_data)
  
}

##################################################
# Combine all methods results
##################################################

output_list_total =  output_list_total[!is.na(output_list_total[,1]),]
output_list_total_DKL = output_list_total_DKL[!is.na(output_list_total_DKL[,1]),]
output_list_total_DKL_default = output_list_total_DKL_default[!is.na(output_list_total_DKL_default[,1]),]
output_list_total_VNNGP = output_list_total_VNNGP[!is.na(output_list_total_VNNGP[,1]),]

list_of_dfs = list(unique(output_list_total_DKL[, c("t", "n_train")]), 
                   unique(output_list_total_DKL_default[, c("t", "n_train")]), 
                   unique(output_list_total[, c("t", "n_train")]),
                   unique(output_list_total_VNNGP[, c("t", "n_train")]))

intersect_rows = Reduce(function(x, y) merge(x, y, by = c("t", "n_train")), list_of_dfs)

colnames(output_list_total_DKL) = c("t","n","index","DKL_w_pred","DKL_w_var","DKL_y_pred","DKL_y_var","n_train")
colnames(output_list_total_DKL_default) = c("t","n","index","DKL_default_w_pred","DKL_default_w_var","DKL_default_y_pred","DKL_default_y_var","n_train")
colnames(output_list_total_VNNGP) = c("t","n","index","VNNGP_w_pred","VNNGP_w_var","VNNGP_y_pred","VNNGP_y_var","n_train")

df1 = merge(output_list_total, intersect_rows, by = c("t", "n_train"))
df2 = merge(output_list_total_DKL[,c("t", "n_train", "index","DKL_w_pred","DKL_w_var","DKL_y_pred","DKL_y_var")], intersect_rows, by = c("t", "n_train"))
df3 = merge(output_list_total_DKL_default[,c("t", "n_train", "index", "DKL_default_w_pred","DKL_default_w_var","DKL_default_y_pred","DKL_default_y_var")], intersect_rows, by = c("t", "n_train"))
df4 = merge(output_list_total_VNNGP[,c("t", "n_train", "index", "VNNGP_w_pred","VNNGP_w_var","VNNGP_y_pred","VNNGP_y_var")], intersect_rows, by = c("t", "n_train"))

dfs_to_merge = list(df1, df2, df3,df4)

output_list_total_sub = Reduce(function(x, y) merge(x, y, by = c("t", "n_train", "index")), dfs_to_merge)

third_elements = c("interval_score", "crps","MSE", "coverage")
second_elements = c("MFA","MFA.LR","spVB-NNGP","spVB-NNGP-joint",
                    "spNNGP")

combination_w = apply(expand.grid("w", second_elements, third_elements), 1, function(x) paste(x, collapse = "_"))
combination_y = apply(expand.grid("y", second_elements, third_elements), 1, function(x) paste(x, collapse = "_"))


colnames(output_list_t) = c("t","n",
                            combination_w,combination_y)

output_list_t = output_list_t[!is.na(output_list_t[,1]),]
output_list_t_DKL = output_list_t_DKL[!is.na(output_list_t_DKL[,1]),] %>% as.matrix()
output_list_t_DKL_default = output_list_t_DKL_default[!is.na(output_list_t_DKL_default[,1]),] %>% as.matrix()
output_list_t_VNNGP = output_list_t_VNNGP[!is.na(output_list_t_VNNGP[,1]),]  %>% as.matrix()

methods = c("DKL", "DKL_default", "VNNGP")

MSE_long = lapply(methods, function(m) {
  output_list_total_sub %>%
    group_by(t, n_train) %>%
    summarise(
      MSE_w = mean((.data[[paste0(m, "_w_pred")]] - w_test)^2, na.rm = TRUE),
      MSE_y = mean((.data[[paste0(m, "_y_pred")]] - y_test)^2, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(method = m)
}) %>%
  bind_rows()

MSE_wide = MSE_long %>%
  pivot_wider(
    id_cols = c(t, n_train),
    names_from = method,
    values_from = c(MSE_w, MSE_y),
    names_glue = "{.value}_{method}_MSE"
  )

colnames(MSE_wide) = gsub("^MSE_w_", "w_", colnames(MSE_wide))
colnames(MSE_wide) = gsub("^MSE_y_", "y_", colnames(MSE_wide))
colnames(MSE_wide) = c("t","n","w_DKL_MSE","w_DKL_default_MSE","w_VNNGP_MSE","y_DKL_MSE","y_DKL_default_MSE","y_VNNGP_MSE")

output_list_t_VNNGP = as.data.frame(output_list_t_VNNGP)
colnames(output_list_t_VNNGP) = c("t","n","w_VNNGP_coverage","w_VNNGP_interval_score","w_VNNGP_crps",
                                  "y_VNNGP_coverage","y_VNNGP_interval_score","y_VNNGP_crps")
output_list_t_VNNGP = left_join(output_list_t_VNNGP, MSE_wide %>% select(t, n, w_VNNGP_MSE, y_VNNGP_MSE), by = c("t", "n"))
output_list_t_VNNGP = na.omit(output_list_t_VNNGP)

output_list_t_DKL = as.data.frame(output_list_t_DKL)
colnames(output_list_t_DKL) = c("t","n","w_DKL_coverage","w_DKL_interval_score","w_DKL_crps",
                                "y_DKL_coverage","y_DKL_interval_score","y_DKL_crps")
output_list_t_DKL = left_join(output_list_t_DKL, MSE_wide %>% select(t, n, w_DKL_MSE, y_DKL_MSE), by = c("t", "n"))
output_list_t_DKL = na.omit(output_list_t_DKL)

output_list_t_DKL_default = as.data.frame(output_list_t_DKL_default)
colnames(output_list_t_DKL_default) = c("t","n","w_DKL_default_coverage","w_DKL_default_interval_score","w_DKL_default_crps",
                                        "y_DKL_default_coverage","y_DKL_default_interval_score","y_DKL_default_crps")
output_list_t_DKL_default = left_join(output_list_t_DKL_default, MSE_wide %>% select(t, n, w_DKL_default_MSE, y_DKL_default_MSE), by = c("t", "n"))
output_list_t_DKL_default = na.omit(output_list_t_DKL_default)


output_list_t_combine = list(output_list_t, output_list_t_DKL, 
                             output_list_t_DKL_default,output_list_t_VNNGP)

intersect_rows = Reduce(function(x, y) merge(x, y, by = c("t", "n")), output_list_t_combine)

merged_data_frames = lapply(output_list_t_combine, function(df) {
  merge(df, intersect_rows, by = c("t", "n"))
})

output_list_mean = Reduce(function(x, y) merge(x, y, by = c("t", "n")), output_list_t_combine)


output_list_mean = output_list_mean[is.na(output_list_mean$n)==F,]

##################################################
# Figure J1 and J2
##################################################
method_colors = c(
  "spVB-MFA" = "#b4d9b8", 
  "spVB-MFA.mb" = "#66c2a5",
  "spVB-MFA-LR" = "#00bfc4",
  "VNNGP" = "#01665e",
  "spVB-NNGP m=1" = "#5e3a8c",  
  "spVB-NNGP.mb m=1" = "#b39ddb", 
  "spVB-NNGP" = "#2171b5",
  "spVB-NNGP-joint" = "#264778",
  "spVB-NNGP.mb m=3" = "#6baed6",
  "spVB-NNGP m=5" = "#3b4994",
  "spVB-NNGP.mb m=5" = "#7986cb",
  "spNNGP" = "#d7301f",
  "DKLGP" = "#e6ab02",
  "DKLGP-default" = "#a6761d"
)


custom_labeller = function(values) {
  sapply(values, function(val) {
    n_train_val = ifelse(val == "100", "1000", ifelse(val == "500", "5000", "10000"))
    paste("n train =", n_train_val, "\nn test =", val)
  })
}

create_plot = function(data_subset, metric_name, metric) {
  # Remove the prefix from Method names and set the levels
  data_subset$Method = sub("w_|y_", "", data_subset$Method)
  data_subset$Method = factor(data_subset$Method, 
                              levels = paste(methods_name,metric,sep="_"),
                              labels = c("spVB-MFA","spVB-MFA-LR","VNNGP","spVB-NNGP","spVB-NNGP-joint",
                                         "DKLGP-default","DKLGP","spNNGP"))
  
  p = ggplot(data_subset, aes(x = Method, y = Score, fill = Method)) + 
    geom_violin(trim = FALSE) +
    stat_summary(fun = mean, geom = "point", shape = 95, size = 4, color = "black") +
    facet_wrap(~`n test`, labeller = as_labeller(custom_labeller), scales = "free_y") +
    theme_minimal() +
    theme(legend.position = "none", strip.text = element_text(size = 10)) +
    labs(x = "Methods", y = paste(gsub('[_\n]', ' ', metric),"for",gsub('[_\n]', '', prefix))) +
    scale_fill_manual(values = method_colors) +
    theme(axis.title.x = element_text(size = 12), 
          axis.title.y = element_text(size = 12),
          axis.text.x = element_text(size = 9,angle = 45, hjust = 1), 
          axis.text.y = element_text(size = 9))
  
  if (metric == "coverage") {
    p = p + geom_hline(yintercept = 0.95, linetype = "dashed", color = "red")
  }else if (metric %in% c("MSE", "interval_score","crps")) {
    min_means = data_subset %>%
      group_by(`n test`,Method) %>%
      summarize(min_mean = min(mean(Score, na.rm = TRUE)), .groups = 'drop') %>%
      group_by(`n test`) %>%
      summarize(minmean = min(min_mean), .groups = 'drop') 
    p = p + geom_hline(data = min_means, aes(yintercept = minmean), linetype = "dotted", color = "blue")
    
  }
  return(p)
}


plots = list()
metrics = c("interval_score","crps","coverage","MSE")
prefixes = c("y_")
methods_name = c("MFA","MFA.LR","VNNGP","spVB-NNGP","spVB-NNGP-joint","DKL_default","DKL","spNNGP")

select_w = apply(expand.grid("w", metrics, methods_name), 1, function(x) paste(x, collapse = "_"))
select_y = apply(expand.grid("y", metrics, methods_name), 1, function(x) paste(x, collapse = "_"))

for (prefix in prefixes) {
  for (metric in metrics) {
    cols_to_select = grep(paste0(prefix, ".*", metric), names(output_list_mean), value = TRUE)
    #cols_to_select = apply(expand.grid(prefix, methods_name, "_",metric), 1, function(x) paste(x, collapse = ""))
    data_subset = pivot_longer(output_list_mean, 
                               cols = all_of(cols_to_select), 
                               names_to = "Method", 
                               values_to = "Score") %>%     
      mutate(n = as.factor(n)) %>%
      mutate(`n test` = case_when(
        n == "1000" ~ "100",
        n == "5000" ~ "500",
        n == "10000" ~ "1000",
        TRUE ~ as.character(n)  # default case if none of the above conditions are met
      )) %>%
      mutate(`n train` = case_when(
        `n test` == "100" ~ "1000",
        `n test` == "500" ~ "5000",
        `n test` == "1000" ~ "10000"
      ))
    data_subset$`n test` = factor(data_subset$`n test`, levels = c("100", "500", "1000"))
    
    plots[[paste(prefix, metric, sep="")]] = create_plot(data_subset, paste(prefix, metric, sep=""), metric)
  }
}

plots_with_titles = lapply(names(plots), function(name) {
  plot = plots[[name]]
  plot + theme(plot.title = element_text(hjust = 0.5))
})

grid.arrange(grobs = plots_with_titles, ncol = 2)
metrics_y = arrangeGrob(grobs = plots_with_titles, ncol = 2)

ggsave(file.path(base_dir, "figures","pred_summary_y.pdf"), plot = metrics_y,
       width = 14, height = 7.5, units = "in", 
       device = cairo_pdf)

plots = list()
metrics = c("interval_score","crps","coverage","MSE")
prefixes = c("w_")
methods_name = c("MFA","MFA.LR","VNNGP","spVB-NNGP","spVB-NNGP-joint","DKL_default","DKL","spNNGP")

select_w = apply(expand.grid("w", metrics, methods_name), 1, function(x) paste(x, collapse = "_"))
select_y = apply(expand.grid("y", metrics, methods_name), 1, function(x) paste(x, collapse = "_"))

for (prefix in prefixes) {
  for (metric in metrics) {
    cols_to_select = grep(paste0(prefix, ".*", metric), names(output_list_mean), value = TRUE)
    #cols_to_select = apply(expand.grid(prefix, methods_name, "_",metric), 1, function(x) paste(x, collapse = ""))
    data_subset = pivot_longer(output_list_mean, 
                               cols = all_of(cols_to_select), 
                               names_to = "Method", 
                               values_to = "Score") %>%     
      mutate(n = as.factor(n)) %>%
      mutate(`n test` = case_when(
        n == "1000" ~ "100",
        n == "5000" ~ "500",
        n == "10000" ~ "1000",
        TRUE ~ as.character(n)  # default case if none of the above conditions are met
      )) %>%
      mutate(`n train` = case_when(
        `n test` == "100" ~ "1000",
        `n test` == "500" ~ "5000",
        `n test` == "1000" ~ "10000"
      ))
    data_subset$`n test` = factor(data_subset$`n test`, levels = c("100", "500", "1000"))
    
    plots[[paste(prefix, metric, sep="")]] = create_plot(data_subset, paste(prefix, metric, sep=""), metric)
  }
}

plots_with_titles = lapply(names(plots), function(name) {
  plot = plots[[name]]
  plot + theme(plot.title = element_text(hjust = 0.5))
})

# Use grid.arrange to arrange your plots with titles
grid.arrange(grobs = plots_with_titles, ncol = 2)

metrics_w = arrangeGrob(grobs = plots_with_titles, ncol = 2)

ggsave(file.path(base_dir, "figures","pred_summary_w.pdf"), , plot = metrics_w,
       width = 14, height = 7.5, units = "in", 
       device = cairo_pdf)

