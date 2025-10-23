# DAC44-Comparison-Metrics

## Version Guide
### Min time
* `v01_gan`: simultaneous training of G and D
* `v02a_d_train`: independent d training (NUB data loader)
* `v02a_d_test`: independent d test (NUB data loader)
* `v02a_g_train`: independent G train (NUB data loader)
* `v02b_d_train`: independent d training (RVC data loader)
* `v02b_d_test`: independent d test (RVC data loader)
* `v02b_g_train`: independent G train (RVC data loader)

### LTI
* `v01_gan`: simultaneous training of G and D
* `v02_gan_twin`: simultaneous training of G and D with twin D
* `v03_d_train`: D training, only, for LTI trajectories
* `v03_g_train`: G training with rules D, only, for straight lines
* `v04_g_train`: G training with rules D and pre-trained D

