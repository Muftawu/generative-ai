### Experiment: β = 0

- Setting β to zero removed KL regularization while retaining the VAE encoder/reparameterization architecture. 
The model produced a more expressive latent representation and substantially improved reconstruction quality. 
This indicates that the KL regularization imposed a meaningful constraint on the latent representation under the current dataset and architecture. 
However, because the latent distribution is no longer encouraged to match the standard normal prior, the resulting model cannot yet be assumed to retain the desirable generative properties of a standard VAE.

### Initial experimentation 
| β     | Val Recon | Val KL     | Active dims | μ std               | logvar std           | Random generation |
|------:|----------:|-----------:|------------:|--------------------:|---------------------:|-------------------|
| 0     | 0.004330  | 711.282308 | 128         | 1.6062872409820557  | 1.4628241062164307   | good              |
| 0.001 | 0.013067  | 2.269408   | 128         | 0.1781105250120163  | 0.0588708333671093   | poor              |
| 0.005 | 0.016304  | 0.083329   | 0           | 0.03862619400024414 | 0.009469778276979923 | poor              |
| 0.01  | ? | ? | ? | ? | ? | ? |
| 0.02  | ? | ? | ? | ? | ? | ? |
| 0.05  | ? | ? | ? | ? | ? | ? |
| 0.1   | ? | ? | ? | ? | ? | ? |

### Zooming in on actual useful beta values (from 0.000 to 0.005)
| β       | Val Recon   | Val KL     | μ mean     | μ std        | logvar mean     | logvar std    | Active dims | Random generation |
|--------:|------------:|-----------:|-----------:|-------------:|----------------:|--------------:|------------:|-------------------|
| 0       | 0.00433     | 711        | ...        | 1.606        | ...             | 1.463         | 128         | Good              |
| .00005  | ?           | ?          | ?          | ?            | ?               | ?             | ?           | ?                 |
| .0001   | ?           | ?          | ?          | ?            | ?               | ?             | ?           | ?                 |
| .00025  | ?           | ?          | ?          | ?            | ?               | ?             | ?           | ?                 |
| .0005   | ?           | ?          | ?          | ?            | ?               | ?             | ?           | ?                 |
| .00075  | ?           | ?          | ?          | ?            | ?               | ?             | ?           | ?                 |
| .001    | 0.01307     | 2.269      | ...        | .178         | ...             | .059          | 128         | Poor              |
| .005    | 0.01630     | .0833      | ...        | .0386        | ...             | .00947        | 0           | Poor              |
