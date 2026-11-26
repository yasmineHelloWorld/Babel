# Fashion-MNIST dataset

- **Source:** Zalando Research Fashion-MNIST (`https://github.com/zalandoresearch/fashion-mnist`).
- **Rows:** 70,000 grayscale 28x28 clothing images (60,000 train + 10,000 test).
- **Features:** 784 pixel intensities scaled and projected via PCA fit strictly on training rows.
- **Label:** 10 clothing categories (`T-shirt/top` through `Ankle boot`).
- **Usage:** vision-derived rate-accuracy curves at `L = 1` across compact discrete symbol budgets.
- **Limitations:** low-resolution centered silhouettes; PCA features discard fine spatial texture compared with a full CNN.
