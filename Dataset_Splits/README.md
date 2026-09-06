# Annotation label mappings

Annotation labels are encoded as integers in the training, validation, and test CSV files for compatibility with PyTorch data-loading pipeline. The mappings between integers and their corresponding labels are provided below.

### Anatomy (`gnd_ana`)

| Integer | Label | Description |
|:---:|---|---|
| 0 | `HC` | Head circumference plane |
| 1 | `Cerebellum` | Transcerebellar plane |
| 2 | `Lips and nose` | Coronal lips and nose plane |
| 3 | `AC` | Abdominal circumference plane |
| 4 | `Femur` | Femur length plane |
| 5 | `Spine (coronal)` | Spine (coronal) |
| 6 | `Spine (sagittal)` | Spine (sagittal) |
| 7 | `Cardiac` | Cardiac |
| 8 | `Other` | Not an NHS FASP view |

### Magnification (`gnd_mag`)

| Integer | Label | Description |
|:---:|---|---|
| 0 | `good_mag` | Good magnification level |
| 1 | `increase_mag` | Need to increase magnification |
| 2 | `reduce_mag` | Need to reduce magnification |

### Gain (`gnd_gain`)

| Integer | Label | Description |
|:---:|---|---|
| 0 | `good_gain` | Good overall gain |
| 1 | `increase_gain` | Need to increase gain |
| 2 | `reduce_gain` | Need to reduce gain |

### Centering (`gnd_centering`)

| Integer | Label | Description |
|:---:|---|---|
| 0 | `good_centering` | Good region of interest centering |
| 1 | `move_left` | Need to move region of interest to the left of the screen |
| 2 | `move_right` | Need to move region of interest to the right of the screen |
| 3 | `move_up` | Need to move region of interest up |
| 4 | `move_down` | Need to move region of interest down |

### Shadow (`gnd_shadow`)

| Integer | Label | Description |
|:---:|---|---|
| 0 | `no_shadow` | No acoustic shadow, or acoustic shadow is minor and does not obscure region of interest |
| 1 | `shadow` | Acoustic shadow is partially or fully obscuring region of interest |