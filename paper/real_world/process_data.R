library(spNNGP)
library(BRISC)
library(rhdf5)
library(dplyr)
library(ggplot2)

base_dir  = file.path(getwd())

test_code = FALSE # set to TRUE for a quick run on a subset of datasets

data(BCEF)
if(test_code){
  set.seed(1234)
  sub_sample = sample(1:nrow(BCEF),5000)
  BCEF = BCEF[sub_sample,]
}

set.seed(1234)
BCEF = BCEF %>%
  mutate(block = (floor(x/1.75) + floor(y*1.25)) %% 2)
table(BCEF$block)

BCEF.in = BCEF %>% filter(block == 0)
BCEF.out = BCEF %>% filter(block == 1)

s = BCEF.in[,c("x","y")] %>% as.matrix()
ord_sumcoords = BRISC_order(s, order = "Sum_coords")
BCEF.in.ordered = BCEF.in[ord_sumcoords,]

S_ordered = BCEF.in.ordered[,c("x","y")] %>% as.matrix()
y = BCEF.in.ordered[,c("FCH")]
y = y - mean(y)
X = cbind((BCEF.in.ordered[,c("PTC")])) %>% as.matrix()
X = X - mean(X)

s_test = BCEF.out[,c("x","y")] %>% as.matrix()
X_test = cbind((BCEF.out[,c("PTC")])) %>% as.matrix()
X_test = X_test - mean(X_test)
X_test = as.matrix(X_test)
y_test = BCEF.out[,c("FCH")]
y_test = y_test - mean(y_test)

n = length(y)
p = ncol(X)
n_test = length(y_test)

data.path = file.path(base_dir, "data")
if(!dir.exists(data.path)){
  dir.create(data.path)
}

# scale the coords to have comparable range with covariates
# to be prepared for VNNGP and DKLGP
rescale_to_range =  function(x, new_min = -75, new_max = 25) {
  (new_max - new_min) * (x - min(x)) / (max(x) - min(x)) + new_min
}

S_ordered_scaled = cbind(rescale_to_range(S_ordered[,1]),rescale_to_range(S_ordered[,2]))
s_test_scaled = cbind(rescale_to_range(s_test[,1]),rescale_to_range(s_test[,2]))

save.image(file = file.path(data.path,"BCEF_data.RData"))

# save data for python read
h5_file_path =  file.path(data.path, "BCEF_data.h5")
if (file.exists(h5_file_path)) {
  file.remove(h5_file_path)
}
h5createFile(h5_file_path)

# Write each dataset to the HDF5 file
h5write(y, h5_file_path, "y")
h5write(as.matrix(X), h5_file_path, "X")
h5write(as.matrix(S_ordered_scaled), h5_file_path, "S_ordered_scaled")
h5write(y_test, h5_file_path, "y_test")
h5write(as.matrix(X_test), h5_file_path, "X_test")
h5write(as.matrix(s_test_scaled), h5_file_path, "s_test_scaled")

