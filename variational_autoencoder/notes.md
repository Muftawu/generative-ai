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
| β       | Val Recon   | Val KL       | μ mean                 | μ std                  | logvar mean           | logvar std          | Active dims | Random generation |
|--------:|------------:|-------------:|-----------------------:|-----------------------:|----------------------:|--------------------:|------------:|-------------------|
| 0       | 0.00433     | 711.282308   | -0.18989816308021545   | 1.6062872409820557     | -9.456109046936035    | 1.4628241062164307  | 128         | Good              |
| .00005  | 0.006680    | 33.071156    | -0.03984536975622177   | 0.5710898637771606     | -0.5858442187309265   | 0.4015151560306549  | 128         | Good              |
| .0001   | 0.007738    | 22.507580    | -0.027332330122590065  | 0.4596284329891205     | -0.3484134078025818   | 0.3042070269584656  | 128         | Good              |
| .00025  | 0.009964    | 8.402109     | -0.012964179739356041  | 0.43022555112838745    | -0.2816421389579773   | 0.19608798623085022 | 128         | Poor              |
| .0005   | 0.011352    | 4.692653     | -0.00995946116745472   | 0.35281795263290405    | -0.19833879172801971  | 0.13635027408599854 | 128         | Poor              |
| .00075  | 0.011920    | 3.447045     | -0.0004635326622519642 | 0.3038414418697357     | -0.140031099319458    | 0.09791576117277145 | 128         | Very Poor         |
| .001    | 0.012617    | 2.468291     | 0.007066719233989716   | 0.2327127307653427     | -0.0874517634510994   | 0.07450447231531143 | 128         | Very Poor         |
| .005    | 0.016304    | 0.083329     | -0.000805908814072609  | 0.03862619400024414    | -0.008725986815989017 | 0.009469778276979923| 0           | Very Very Poor    |

### observation
- so for [0.00000, 0.00005, 0.0001] have we have all colors, the model is able to generalize the reconstruction with all white, blue and black colors of the image and the general positions of each sections so quite good.
- for beta values of [0.00025, 0.00075], the reconstruction is very poor, where it losses the blue and subtle black colors but has a generalized reconstruction for the white sections of the image 
- for beta values of [0.001, 0.005] also have very very poor reconstruction as it results in just a white blob generalized across the reconstructed images so not much info embedded or learned by the encoder
