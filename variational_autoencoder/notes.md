### Experiment: β = 0

- Setting β to zero removed KL regularization while retaining the VAE encoder/reparameterization architecture. 
The model produced a more expressive latent representation and substantially improved reconstruction quality. 
This indicates that the KL regularization imposed a meaningful constraint on the latent representation under the current dataset and architecture. 
However, because the latent distribution is no longer encouraged to match the standard normal prior, the resulting model cannot yet be assumed to retain the desirable generative properties of a standard VAE.

| β     | Val Recon | Val KL     | Active dims | μ std               | logvar std          | Random generation |
|------:|----------:|-----------:|------------:|--------------------:|--------------------:|-------------------|
| 0     | 0.004330  | 711.282308 | 128         | 1.6062872409820557  | 1.4628241062164307  | good              |
| 0.001 | 0.013067  | 2.269408   | 128         | 0.1781105250120163  | 0.0588708333671093  | poor              |
| 0.005 | ? | ? | ? | ? | ? | ? |
| 0.01  | ? | ? | ? | ? | ? | ? |
| 0.02  | ? | ? | ? | ? | ? | ? |
| 0.05  | ? | ? | ? | ? | ? | ? |
| 0.1   | ? | ? | ? | ? | ? | ? |
