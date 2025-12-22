########################################
############## load package ############
########################################
library(dplyr)
library(ggplot2)
library(viridis)
library(patchwork)
library(grid)
library(cowplot)
library(dplyr)
library(tidyverse)
library(ggrastr)
library(tidyr)
library(scales)

########################################
############## Read data ###############
########################################
base_dir  = file.path(getwd())
data.path = file.path(base_dir, "data")
output.path = file.path(base_dir, "output")
fig.path = file.path(base_dir, "fig")
load(file.path(data.path,"BCEF_data.RData"))

if(!dir.exists(fig.path)){
  dir.create(fig.path)
}

########################################
############## Figure 4 ################
########################################
FCH_fig = BCEF %>% ggplot() +
  geom_point(aes(x=x,y=y,col=FCH),size = 0.5) + 
  scale_color_viridis_c(name = "Forest Canopy Height (FCH)", direction = -1) +
  labs(x = "Easting (km)", y = "Northing (km)") +
  theme_minimal()+
  theme(
    legend.position   = "bottom",
    legend.title.align = 0.5
  )

FCH_fig$layers[[1]] = rasterise(FCH_fig$layers[[1]], dpi = 300)

ggsave(file.path(fig.path,"FCH_raster.pdf"), plot = FCH_fig,
       width = 5.1, height = 5.25, units = "in", 
       device = cairo_pdf)

PTC_fig = BCEF %>% ggplot() +
  geom_point(aes(x=x,y=y,col=PTC),size = 0.5) + 
  scale_color_viridis_c(option = "plasma", name = "Percent Tree Cover (PTC)", direction = -1) +
  labs(x = "Easting (km)", y = "Northing (km)") +
  theme_minimal()+
  theme(
    legend.position   = "bottom",
    legend.title.align = 0.5
  )

PTC_fig$layers[[1]] = rasterise(PTC_fig$layers[[1]], dpi = 300)

ggsave(file.path(fig.path,"PTC_raster.pdf"), plot = PTC_fig,
       width = 5.1, height = 5.25, units = "in", 
       device = cairo_pdf)

########################################
############# Figure 5 #################
########################################
load(file.path(output.path, "MFA_fit.RData"))
load(file.path(output.path, "MFA_LR_fit.RData"))
load(file.path(output.path, "NNGP_ind_fit.RData"))
load(file.path(output.path, "NNGP_joint_fit.RData"))
load(file.path(output.path, "spNNGP_fit.RData"))
load(file.path(output.path, "VNNGP_fit.RData"))
load(file.path(output.path, "DKLGP_fit.RData"))

df_all_mean = data.frame(
  spNNGP_mean = spNNGP_summary$w_mean,
  MFA_mean = MFA_summary$w_mean,
  MFA_LR_mean = MFA_LR_summary$w_mean,
  NNGP_mean = NNGP_ind_summary$w_mean,
  NNGP_joint_mean = NNGP_joint_summary$w_mean,
  VNNGP_mean = VNNGP_summary$w_mean,
  DKLGP_mean = DKLGP_summary$w_mean
)

output_list_total_long = df_all_mean %>%
  select(spNNGP_mean, MFA_mean, MFA_LR_mean, NNGP_mean, NNGP_joint_mean, VNNGP_mean, DKLGP_mean) %>%
  pivot_longer(cols = -spNNGP_mean, names_to = "model", values_to = "estimated_w") %>%
  filter(is.finite(spNNGP_mean), is.finite(estimated_w)) 

output_list_total_long$model = factor(output_list_total_long$model,
                                      levels = c("MFA_mean", "MFA_LR_mean", "VNNGP_mean", "NNGP_mean", "NNGP_joint_mean",  "DKLGP_mean"),
                                      labels = c("spVB-MFA", "spVB-MFA-LR", "VNNGP", "spVB-NNGP", "spVB-NNGP-joint", "DKLGP-default"))

p1 = ggplot(output_list_total_long) + 
  geom_point(aes(spNNGP_mean, estimated_w), alpha = 0.3) +
  facet_grid(~model) + 
  geom_abline(intercept = 0, slope = 1, color = "red", size = 0.3) + 
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, size = 11),
    strip.text = element_text(size = 12),
    axis.title        = element_text(size = 12), 
    aspect.ratio = 1
  ) +
  labs(
    x = "MCMC Estimated Posterior Mean for w (spNNGP)",
    y = "Approximated Posterior Mean for w"
  )

p1$layers[[1]] = rasterise(p1$layers[[1]], dpi = 300)
ggsave(file.path(fig.path,"real_w_mean_sub_raster.pdf"), plot = p1,
       width = 14.97, height = 3.89, units = "in", 
       dpi = 300, device = cairo_pdf)


df_all_var = data.frame(
  spNNGP_var = spNNGP_summary$w_var,
  MFA_var = MFA_summary$w_var,
  MFA_LR_var = MFA_LR_summary$w_var,
  NNGP_var = NNGP_ind_summary$w_var,
  NNGP_joint_var = NNGP_joint_summary$w_var,
  VNNGP_var = VNNGP_summary$w_var,
  DKLGP_var = DKLGP_summary$w_var
)

# Reshape to long format
output_list_var_long = df_all_var %>%
  select(spNNGP_var, MFA_var, MFA_LR_var, NNGP_var, NNGP_joint_var, VNNGP_var, DKLGP_var) %>%
  pivot_longer(cols = -spNNGP_var, names_to = "model", values_to = "estimated_var") %>%
  filter(is.finite(spNNGP_var), is.finite(estimated_var))  

output_list_var_long$model = factor(output_list_var_long$model,
                                    levels = c("MFA_var", "MFA_LR_var", "VNNGP_var", "NNGP_var", "NNGP_joint_var", "DKLGP_var"),
                                    labels = c("spVB-MFA", "spVB-MFA-LR", "VNNGP", "spVB-NNGP", "spVB-NNGP-joint",  "DKLGP-default"))

p2 = ggplot(output_list_var_long) + 
  geom_point(aes(spNNGP_var, estimated_var), alpha = 0.3) +
  facet_grid(~model) + 
  geom_abline(intercept = 0, slope = 1, color = "red", size = 0.3) + 
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, size = 11),
    strip.text = element_text(size = 12),
    axis.title        = element_text(size = 12), 
    aspect.ratio = 1
  ) +
  labs(
    x = "MCMC Estimated Posterior Variance for w (spNNGP)",
    y = "Approximated Posterior Variance for w"
  )

p2$layers[[1]] = rasterise(p2$layers[[1]], dpi = 300)
ggsave(file.path(fig.path,"real_w_var_sub_raster.pdf"), plot = p2,
       width = 14.97, height = 3.89, units = "in", 
       dpi = 300, device = cairo_pdf)


########################################
############# Figure 6 #################
########################################
load(file.path(output.path, "MFA_predict_w_y.RData"))
load(file.path(output.path, "MFA_LR_predict_w_y.RData"))
load(file.path(output.path, "NNGP_ind_predict_w_y.RData"))
load(file.path(output.path, "NNGP_joint_predict_w_y.RData"))
load(file.path(output.path, "spNNGP_predict_w_y.RData"))
load(file.path(output.path, "VNNGP_predict_w_y.RData"))
load(file.path(output.path, "DKLGP_predict_w_y.RData"))

y_pred= data.frame(s_test,
                   y_test,
                   y_var_pred_spNNGP,
                   y_var_pred_NNGP_joint,
                   y_var_pred_NNGP,
                   y_var_pred_MFA_LR,
                   y_var_pred_MFA,
                   y_var_pred_VNNGP,
                   y_var_pred_DKLGP)

baseline = y_pred %>%
  select(x, y, baseline = y_var_pred_spNNGP)

y_pred_ratio = y_pred %>%
  select(x, y,
         MFA = y_var_pred_MFA,
         MFA_LR = y_var_pred_MFA_LR,
         NNGP = y_var_pred_NNGP,
         NNGP_joint = y_var_pred_NNGP_joint,
         VNNGP = y_var_pred_VNNGP,
         DKLGP = y_var_pred_DKLGP) %>%
  pivot_longer(
    cols = -c(x, y),
    names_to = "method",
    values_to = "var_pred"
  ) %>%
  left_join(baseline, by = c("x", "y")) %>%
  mutate(
    method = recode(method,
                    "MFA" = "spVB-MFA",
                    "MFA_LR" = "spVB-MFA-LR",
                    "NNGP" = "spVB-NNGP",
                    "NNGP_joint" = "spVB-NNGP-joint",
                    "VNNGP" = "VNNGP",
                    "DKLGP" = "DKLGP-default"
    ),
    var_diff = var_pred / baseline
  )

summary_stats = y_pred_ratio %>%
  group_by(method) %>%
  summarise(
    ratio = mean(var_diff, na.rm = TRUE),
    .groups = "drop"
  )

y_pred_ratio = y_pred_ratio %>%
  left_join(summary_stats, by = "method") %>%
  mutate(
    facet_label = paste0(method, "\nRatio = ", round(ratio, 2))
  )

y_pred_ratio$method = factor(y_pred_ratio$method,
                             levels = c("spVB-MFA", "spVB-MFA-LR", "VNNGP",
                                        "spVB-NNGP", "spVB-NNGP-joint",  
                                        "DKLGP-default"))

p3 = ggplot(y_pred_ratio, aes(x = x, y = y, col = var_diff)) +
  geom_point(size = 0.5) +
  facet_wrap(~ method, nrow = 1) +
  scale_color_gradient2(
    name = "Variance Ratio\nvs spNNGP",
    midpoint = 1,  # center at 1
    low = "#2166ac", mid = "white", high = "#b2182b",
    limits = c(0.25, 2.5),  # adjust based on your range
    oob = squish
  )+
  theme_minimal() +
  theme(
    legend.position = "bottom",
    strip.text = element_text(size = 12)
  ) +
  labs(
    x = "Easting (km)", y = "Northing (km)"
  )

p3$layers[[1]] = rasterise(p3$layers[[1]], dpi = 300)
ggsave(file.path(fig.path,"real_y_pred_var_raster.pdf"), plot = p3,
       width = 14.97, height = 3.89, units = "in", 
       dpi = 300, device = cairo_pdf)

########################################
############# Figure 7 #################
########################################
dataset = data.frame(
  methods = c("spVB-MFA", "spVB-MFA-LR", "spVB-NNGP", "spVB-NNGP-joint", "spNNGP", 
              "VNNGP","DKLGP-default"),
  time = c(MFA_summary$run_time[3] %>% as.numeric(),
           MFA_LR_summary$run_time[3] %>% as.numeric(),
           NNGP_ind_summary$run_time[3] %>% as.numeric(),
           NNGP_joint_summary$run_time[3] %>% as.numeric(),
           spNNGP_summary$run_time[3] %>% as.numeric(),
           VNNGP_summary$run_time %>% as.numeric(),
           DKLGP_summary$run_time %>% as.numeric())
)

dataset$methods = factor(dataset$methods, levels = c(
  "spVB-MFA", "spVB-MFA-LR", "VNNGP", 
  "spVB-NNGP", "spVB-NNGP-joint", 
  "DKLGP", "DKLGP-default", "spNNGP"
))

# Define your color palette
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
  "DKLGP-default" = "#a6761d",
  "INFVB" = "darksalmon"
)


time_original = ggplot(dataset, 
                       aes(x = methods, y = time, fill = methods)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(time), y = time + 2000), size = 4, face = "bold") +
  scale_fill_manual(values = method_colors) +
  labs(
    x = "Method",
    y = "Running Time (seconds)"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.title = element_text(size = 12),       # Bold and larger axis titles
    axis.text = element_text(size = 11),     
    axis.text.x = element_text(angle = 30, hjust = 1)# Bold and larger axis tick labels
  )

time_original

ggsave(file.path(fig.path,"time_original.pdf"), plot = time_original,
       width = 6.35, height = 4.3, units = "in", 
       device = cairo_pdf)

########################################
############# Table 4 ##################
########################################
make_method_row = function(method) {
  env = parent.frame()
  
  crps      = get(paste0(method, "_crps"),      envir = env)
  is        = get(paste0(method, "_is"),        envir = env)
  mse       = get(paste0(method, "_mse"),       envir = env)
  coverage  = get(paste0(method, "_coverage"),  envir = env)
  summary   = get(paste0(method, "_summary"),   envir = env)
  
  f4 = function(x) formatC(x, format = "f", digits = 4)
  f2 = function(x) formatC(x, format = "f", digits = 2)
  f1 = function(x) formatC(x, format = "f", digits = 1)
  
  ## -------- beta ----------
  if (is.null(summary$beta_quantile)) {
    beta_ci = "-"
  } else {
    beta_ci = paste0(
      f4(summary$beta_mean),
      " (",
      f4(summary$beta_quantile[1]), ",",
      f4(summary$beta_quantile[2]),
      ")"
    )
  }
  
  ## -------- theta (sigmasq, tausq) ----------
  if (is.null(summary$theta_quantile)) {
    theta1_ci = f1(as.numeric(summary$theta_mean$sigmasq))
    theta2_ci = f1(as.numeric(summary$theta_mean$tausq))
  } else {
    # sigmasq 
    theta1_ci = paste0(
      f1(summary$theta_mean[1]), " (",
      f1(summary$theta_quantile[1, 1]), ",",
      f1(summary$theta_quantile[2, 1]),
      ")"
    )
    
    # tausq 
    theta2_ci = paste0(
      f2(summary$theta_mean[2]), " (",
      f2(summary$theta_quantile[1, 2]), ",",
      f2(summary$theta_quantile[2, 2]),
      ")"
    )
  }
  
  ## -------- method label map ----------
  label_map = c(
    MFA        = "spVB-MFA",
    MFA_LR     = "spVB-MFA-LR",
    NNGP_ind   = "spVB-NNGP",
    NNGP_joint = "spVB-NNGP-joint",
    spNNGP     = "spNNGP",
    VNNGP      = "VNNGP",
    DKLGP      = "DKLGP-default"
  )
  
  pretty_method = if (method %in% names(label_map)) {
    label_map[[method]]
  } else {
    method
  }
  
  data.frame(
    Method   = pretty_method, 
    beta     = beta_ci,
    sigmasq  = theta1_ci,
    tausq    = theta2_ci,
    CRPS     = formatC(crps,         format = "f", digits = 2),
    IS       = formatC(is,           format = "f", digits = 3),
    MSE      = formatC(mse,          format = "f", digits = 1),
    Coverage = formatC(coverage * 100, format = "f", digits = 1),
    check.names = FALSE,
    row.names   = NULL
  )
}


method_table = rbind(make_method_row("MFA"), 
      make_method_row("MFA_LR"), 
      make_method_row("VNNGP"),
      make_method_row("NNGP_ind"),
      make_method_row("NNGP_joint"),
      make_method_row("DKLGP"),
      make_method_row("spNNGP"))

write.csv(method_table, file.path(fig.path,"BCEF_summary.csv"), row.names = FALSE)

