### Experiment: β = 0

- Setting β to zero removed KL regularization while retaining the VAE encoder/reparameterization architecture. 
The model produced a more expressive latent representation and substantially improved reconstruction quality. 
This indicates that the KL regularization imposed a meaningful constraint on the latent representation under the current dataset and architecture. 
However, because the latent distribution is no longer encouraged to match the standard normal prior, the resulting model cannot yet be assumed to retain the desirable generative properties of a standard VAE.

### Initial experimentation [RUN 1] 
| β     | Val Recon | Val KL     | Active dims | μ std               | logvar std           | Random generation |
|------:|----------:|-----------:|------------:|--------------------:|---------------------:|-------------------|
| 0     | 0.004330  | 711.282308 | 128         | 1.6062872409820557  | 1.4628241062164307   | good              |
| 0.001 | 0.013067  | 2.269408   | 128         | 0.1781105250120163  | 0.0588708333671093   | poor              |
| 0.005 | 0.016304  | 0.083329   | 0           | 0.03862619400024414 | 0.009469778276979923 | poor              |
| 0.01  | ? | ? | ? | ? | ? | ? |
| 0.02  | ? | ? | ? | ? | ? | ? |
| 0.05  | ? | ? | ? | ? | ? | ? |
| 0.1   | ? | ? | ? | ? | ? | ? |

### Zooming in on actual useful beta values (from 0.000 to 0.005) [RUN 2]
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

### Detailed Description [RUN 2]
- At β=0, the model learned to use the latent space aggressively to preserve visual information, producing its strongest reconstructions, but the latent distribution was highly irregular.
- At β=.00005–.0001, the model learned to retain substantial visual information while beginning to regularize the latent space. All 128 dimensions remained active and random generation remained useful in your tests.
- Around β=.00025 and above, the KL constraint increasingly outweighed the model's ability to preserve less dominant visual information, such as the blue/brown regions you observed.
- By β=.005, the latent representation had become so close to the prior that the decoder received too little image-specific information, producing generalized reconstructions.


### Second zooming in finer model response for beta values (0.00000 to 0.00025) [RUN 3]
| β       | Val Recon   | Val KL       | μ mean                 | μ std                  | logvar mean           | logvar std          | Var Mean             | Var Std               | Act dims | Random Gen    | Visual Recon      |
|--------:|------------:|-------------:|-----------------------:|-----------------------:|----------------------:|--------------------:|---------------------:|----------------------:|---------:|--------------:|-------------------|
| 0       | 0.004330    | 711.282308   | -0.18989816308021545   | 1.6062872409820557     | -9.456109046936035    | 1.4628241062164307  | 0.000320866412948817 | 0.0015134125715121627 |128       | Poor          | Poor              |
| .00001  | 0.005178    | 99.228432    | -0.05689043924212456   | 0.80110764503479       | -1.6778758764266968   | 0.5328776836395264  | 0.21442142128944397  | 0.1152794361114502    |128       | Poor          | Poor              |
| .000025 | 0.005881    | 54.366108    | -0.05361516401171684   | 0.6803540587425232     | -0.9438456296920776   | 0.4724269211292267  | 0.429471492767334    | 0.17773893475532532   |128       | Poor          | Poor              |
| .00005  | 0.006680    | 33.071156    | -0.03984536975622177   | 0.5710898637771606     | -0.5858442187309265   | 0.4015151560306549  | 0.5953892469406128   | 0.19149410724639893   |128       | Very poor     | Very poor         |
| .000075 | 0.007415    | 24.091866    | -0.040598705410957336  | 0.5037879943847656     | -0.43331179022789     | 0.35143938660621643 | 0.6820114850997925   | 0.1860070824623108    |128       | Very poor     | Better than random|
| .0001   | 0.007961    | 19.070073    | -0.027332330122590065  | 0.4596284329891205     | -0.3484134078025818   | 0.3042070269584656  | 0.7331624031066895   | 0.1728619635105133    |128       | Very Poor     | Better than random|
| .000125 | 0.008310    | 16.311399    | -0.025205332785844803  | 0.4321693181991577     | -0.3039793372154236   | 0.2679586708545685  | 0.760212242603302    | 0.1597907990217209    |128       | Very Poor     | Blurry blob       |
| .00015  | 0.009006    | 12.858442    | -0.039637062698602676  | 0.49818575382232666    | -0.4159316420555115   | 0.2797951102256775  | 0.6839730143547058   | 0.1728149950504303    |128       | Pixelated blob| Poor white blob   |
| .000175 | 0.009347    | 11.096020    | -0.03458011895418167   | 0.48049068450927734    | -0.37227901816368103  | 0.27058833837509155 | 0.7128127217292786   | 0.1739969253540039    |128       | Pixelated blob| Poor white blob   |
| .0002   | 0.009553    | 10.262117    | -0.02488628588616848   | 0.46530237793922424    | -0.3403279781341553   | 0.2495451271533966  | 0.7322098016738892   | 0.16458138823509216   |128       | Pixelated blob| Poor white blob   |
| .00025  | 0.009964    | 8.402109     | -0.000805908814072609  | 0.43022555112838745    | -0.2816421389579773   | 0.19608798623085022 | 0.7682754993438721   | 0.13856589794158936   |128       | Pixelated blob| Poor white blob   |


### Detailed Description [RUN 3]
- For this particular 128-dimensional convolutional VAE, trained on the current 200-image dataset at 64×64 resolution with MSE reconstruction and the current annealing schedule, 
- increasing β causes a clear trade-off between latent regularization and reconstruction quality. 
- Very small β values preserve substantially more image information, while larger β values progressively force the latent representation toward the standard normal prior 
- and eventually produce blurry/pixelated average-like reconstructions.

### for linear interpolation
- take 2 images, A & B, 
- encode them -> μA, μB
- then z0 = μA & z1 = μB 
- then using this range (run over [0.00, 0.25, 0.50, 0.75, 1.00]), gradually interpolate for 
- z(t) = (1-t) μA + t μB 
- result should look more like this 
`
    A
    ↓
    something between A and B
    ↓
    something more like B
    ↓
    B
`

### Latent Dimension Experimentation (testing between 16, 32, 64, 128)
- Lat Dim = 16
    - Val Recon                 = 
    - Val KL                    = 
    - μ mean                    = 
    - μ std                     = 
    - var mean                  = 
    - var std                   = 
    - kl per dimension          = 
    - random generation         =
    - zero latent recon         =
    - μ reconstruction          = 
    - latent interpolation      =



