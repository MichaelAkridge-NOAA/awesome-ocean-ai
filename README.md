# Ocean AI Toolkit 
Collection of tools and resources for marine image/video analysis, annotation, and detection. 

## Table of Contents
- [Models](#models)
- [Identification Resources](#identification-resources)
- [Training Data Sets](#training-data-sets)
- [Image Annotation](#image-annotation)
- [Video Annotation](#video-annotation)
- [AI / ML Backends](#ai--ml-backends)
- [Miscellaneous](#miscellaneous)

---
## Starter Courses 
- https://developers.google.com/machine-learning/crash-course/

## StarterTools 
- https://colab.research.google.com   (Free GPU Access)
- https://notebooklm.google/

## Image and Video Annotation
| Name | Image | Video | Free | Open Source | AI Assisted Annotation | Notes |
|---------------------------------------------------|-------|-------|------|-------------|------------------------|-----------------------------------------------------------------------------------------------|
| [Label Studio](https://github.com/HumanSignal/label-studio) | ✅ | ✅ | ✅ | ✅ | ✅ | Multi-type data labeling and annotation tool with standardized output format. |
| [CVAT](https://github.com/opencv/cvat) | ✅ | ✅ | ✅ | ✅ | ✅ | Open-source tool for video and image annotation. |
| [VIAME](http://www.viametoolkit.org/) | ✅ | ✅ | ✅ | ✅ | ✅ | See [documentation](https://viame.readthedocs.io/en/latest/section_links/documentation_overview.html). |
| [Scalabel](https://github.com/scalabel/scalabel) | ✅ | ✅ | ✅ | ✅ |  | Modern video and image annotation platform. |
| [Tator](https://www.tator.io/) | ✅ | ✅ | ✅ | ✅ |  | See [Tutorials](https://www.tator.io/tutorials) and [Source Code](https://github.com/cvisionai/Tator). |
| [BIIGLE](https://www.biigle.de/) | ✅ | ✅ | ✅ | ✅ |  | Check [GitHub](https://github.com/biigle/). |
| [FishID](https://globalwetlandsproject.org/tools-2/fishid/) | ✅ | ✅ | ✅ | 🔜 |  | Email to request access: fishidglow@outlook.com. |
| [TagLab](https://github.com/cnr-isti-vclab/TagLab/) | ✅ | ❌ | ✅ | ✅ | ✅ |  |
| [VoTT (Archived)](https://github.com/Microsoft/VoTT) | ✅ | ❌ | ✅ | ✅ |  | Officially archived, last update in 2020. |
| [SEAMS](https://github.com/begeospatial/seams-app) | ✅ | ❌ | ✅ | ✅ | ❌ | SEafloor Annotation and Mapping Support Open-source tool for video and image annotation. |
| [labelme](https://github.com/wkentaro/labelme) | ✅ | ❌ | ✅ | ✅ | ❌ | Basic annotation tool for labeling images. |
| [Sebastes](https://repository.library.noaa.gov/view/noaa/11999/noaa_11999_DS1.pdf) | ✅ | ❌ | ✅ | ✅ | ❌ | Used for fish detection. |
| [Squidle+](https://bitbucket.org/ariell/marinedb) | ✅ | ❌ | ✅ | ✅ | ❌ | Marine biology database and annotation tool. |
| [CoralNet](https://coralnet.ucsd.edu/) | ✅ | ❌ | ✅ | ❌ | ✅ | Great for coral species identification. |
| [Coral Point Count with Excel extensions](https://cnso.nova.edu/cpce/index.html) | ✅ | ❌ | ✅ | ❌ | ❌ | Popular for coral reef studies. |
| [BenthoBox](https://benthobox.com) | ✅ | ❌ | ✅ | ❌ | ❌ | Benthic imagery analysis tool. |
| [Supervisely](https://supervise.ly/) | ✅ | ❌ | ❌ | ❌ | ✅ | Paid service, widely used for AI-based labeling. |
| [Deep Sea Spy](https://www.deepseaspy.com) | ✅ | ❌ | ❌ | ❌ | ❌ | Online annotation tool. |
| [Labelbox](https://labelbox.com/) | ✅ | ❌ | ❌ | ❌ | ✅ | Leading platform for AI-assisted labeling. |
| [OFOP](http://www.ofop-by-sams.eu/) | ✅ | ❌ | ❌ | ❌ | ❌ | Oceanographic annotation platform. |
| [RectLabel](https://rectlabel.com/) | ✅ | ❌ | ❌ | ❌ | ❌ | Paid video/image annotation tool for macOS. |
| [SeaGIS](https://www.seagis.com.au/) | ✅ | ❌ | ❌ | ❌ | ❌ | Includes [EventMeasure](https://www.seagis.com.au/event.html) and [TransectMeasure](https://www.seagis.com.au/transect.html). |
| [MBARI Media Management (M3/VARS)](https://mbari-media-management.github.io/) | ❌ | ✅ | ✅ | ✅ | ❌ | Specialized tool for deep-sea video management. |
| [VARS](https://hohonuuli.github.io/vars/) | ❌ | ✅ | ✅ | ✅ | ❌ | Annotation and reference system for underwater video. |
| [video-annotation-tool](https://github.com/video-annotation-project/video-annotation-tool) | ❌ | ✅ | ✅ | ✅ | ❌ | Lightweight, open-source tool. |
| [Digital Fishers](https://www.oceannetworks.ca/learning/get-involved/citizen-science/digital-fishers) | ❌ | ✅ | ❌ | ❌ | ❌ | Citizen science platform for video classification. |
| [SeaTube](http://dmas.uvic.ca/SeaTube) | ❌ | ✅ | ✅ | ❌ | ❌ | Tool for underwater video annotation. |
| [ADELIE](https://www.flotteoceanographique.fr/La-Flotte/Logiciels-embarques/ADELIE) | ❌ | ✅ | ✅ | ❌ | ❌ | Marine research annotation tool. |

---
## AI / ML Frameworks
- [KSO System](https://github.com/ocean-data-factory-sweden/kso) - Koster Seafloor Observatory System is an open-source, citizen science and machine learning approach framework of tools
  - https://subsim.se/ - the Swedish platform for subsea image analysis
- https://github.com/esri/deep-learning-frameworks
---
## AI / ML Backends
| Name | Link | Notes |
|------|------|-------|
| Nuclio | [Nuclio GitHub](https://github.com/nuclio/nuclio) | Serverless AI/ML platform for scalable computing. |
| Label Studio ML Backend | [Label Studio ML](https://github.com/HumanSignal/label-studio-ml-backend) | AI backends for Label Studio. |
| Tator ML Backend | [Tator GitHub](https://github.com/cvisionai/Tator-ML) | Backend system for AI-driven video annotation. |

---

## AI / ML Models
| Name  | Link  | Description |
|-------|-------|-------------|
| FathomNet | [Hugging Face](https://huggingface.co/FathomNet) / [GitHub](https://github.com/fathomnet/models) | Models for marine species detection and analysis. |
| Fish Detector  | [YOLOv8n Grayscale Fish Detector](https://huggingface.co/akridge/yolo8-fish-detector-grayscale) | Grayscale YOLOv8 model trained to detect fish in marine imagery. |
| VIAME Model Zoo | [VIAME Model Zoo](https://github.com/VIAME/VIAME/wiki/Model-Zoo-and-Add-Ons) | Collection of models and add-ons for video and image-based marine analysis. |
| Placeholder | | |
- https://www.zooniverse.org/projects/victorav/spyfish-aotearoa
- https://zenodo.org/records/5539915
- https://github.com/facebookresearch/sam2
- https://github.com/facebookresearch/detr
- https://github.com/ultralytics/ultralytics
- https://www.kaggle.com/models/google/multispecies-whale
- https://coral.ai/models/
- https://github.com/google-coral/example-object-tracker
- Model: https://zenodo.org/records/13589902
---

## Identification Resources
- [Encyclopedia of Life](https://eol.org/)
- [MBARI Deep-Sea Guide](http://dsg.mbari.org)
- [Tree of Life](http://tolweb.org)
- [World Register of Marine Species (WoRMS)](http://www.marinespecies.org/)

---

## Training Data Sets & Data Hubs
- [Hugging Face](https://huggingface.co/)
- [Roboflow](https://roboflow.com/)
- https://roboflow.com/pricing | Example: https://universe.roboflow.com/uwrov-2022-ml-challenge/deepsea-detect--mate-2022-ml-challenge
- https://www.zooniverse.org
- [Atlantis Dataset](https://github.com/smhassanerfani/atlantis)
- [FathomNet](http://fathomnet.org)
- [Fishnet.AI](https://www.fishnet.ai/)
- [FishIDLowViz](https://github.com/slopezmarcano/dataset-fish-detection-low-visibility)
- [FishIDLuderick](https://github.com/globalwetlands/luderick-seagrass)
- [DeepFish](https://alzayats.github.io/DeepFish/)
- [VIAME](https://viame.kitware.com)
- [Croatian Fish Dataset](https://github.com/jaeger-j/FishID/tree/main/datasets)
- Dataset: https://www.gbif.org/dataset/51d0bd32-e215-45ea-a04d-47a474336125
- Mermaid - https://datamermaid.org/
---

## Data Managment 
- https://github.com/csiro-fair/marimba
- https://desktop.arcgis.com/en/arcmap/latest/manage-data/geodatabases/what-is-a-geodatabase.htm

## Apps
- https://github.com/MichaelAkridge-NOAA/Fish-or-No-Fish-Detector
- https://github.com/tkwant/fish-localization
- https://github.com/ShrimpCryptid/deepsea-detector
  - https://universe.roboflow.com/uwrov-2022-ml-challenge/deepsea-detect--mate-2022-ml-challenge

# AI / ML Deep Learning & GIS
- https://github.com/coolzhao/Geo-SAM
- https://github.com/esri/deep-learning-frameworks
- https://doc.arcgis.com/en/pretrained-models/latest/imagery/using-segment-anything-model-sam-.htm
- https://www.arcgis.com/home/item.html?id=9b67b441f29f4ce6810979f5f0667ebe
- https://doc.arcgis.com/en/pretrained-models/latest/imagery/introduction-to-segment-anything-model-sam-.htm

# News, Readings & Journals
- https://www.fisheries.noaa.gov/feature-story/hey-google-find-new-whale-sound?utm_medium=email&utm_source=govdelivery
- https://www.fisheries.noaa.gov/pacific-islands/population-assessments/passive-acoustics-pacific-islands
- https://www.fisheries.noaa.gov/new-england-mid-atlantic/science-data/geospatial-artificial-intelligence-animals
- Harmful algae classifier: https://www.epa.gov/sites/default/files/2018-10/documents/benthic-habs-using-machine-10232018.pdf
- https://wildlife.ai/using-ai-to-identify-new-zealand-fish-project-completion-update/

## Miscellaneous
| Name | Free | Open Source | Notes |
| -- | -- | -- | -- |
| [Norfair](https://github.com/tryolabs/norfair) | ✅ | ✅ | Lightweight Python library for adding real-time multi-object tracking to any detector.|
| [3D Metrics](https://3d-metrics.com/) | ❌ | ❌ | 3D analysis for marine imagery. |
| [Cthulhu](https://github.com/mbari-media-management/cthulhu) | ✅ | ✅ | Video player with remote control API and localization support. |
| [iNaturalist](https://www.inaturalist.org/) | ✅ | ❌ | Biodiversity identification resource. |
| [Matisse](https://www.eso.org/sci/facilities/develop/instruments/matisse.html) | ❌ | ❌ | Tool for spectral analysis. |
| [OpenCV](https://opencv.org/) | ✅ | ✅ | Computer vision library for image/video processing. |
| [Sharktopoda](https://github.com/mbari-org/Sharktopoda) | ✅ | ✅ | Video player with remote control and bounding box support. |
| [FishIDTrackers](https://github.com/slopezmarcano/automated-fish-tracking) | ✅ | ✅ | Fish detection and tracking pipelines with demo datasets. |

# To Review:
- ReefCloud - https://reefcloud.ai/
- https://github.com/HumanSignal/awesome-data-labeling
- https://github.com/mattdawkins/awesome-data-annotation
- https://github.com/cappelletto/awesome-coral-research
- https://github.com/heinsense2/AIO_CaseStudy
- https://github.com/Ahmad-AlShalabi/Fish-detection
- https://github.com/yohmori/Fish-Detector
- https://github.com/patrick-tssn/Streaming-Grounded-SAM-2
- https://github.com/slopezmarcano/automated-fish-tracking
- https://github.com/slopezmarcano/YOLO-Fish
- https://github.com/slopezmarcano/fishid-budgeting
- https://github.com/slopezmarcano/computer-vision-training-the-gambia
- https://github.com/slopezmarcano/dataset-fish-detection-low-visibility
- https://www.zooniverse.org/projects/victorav/the-koster-seafloor-observatory/classify/workflow/23483
- https://snd.se/sv/catalogue/dataset/2022-98-1/1
- https://github.com/wildlifeai/FishDetection-FasterRCNN-project
----------
#### Disclaimer
This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project content is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.

##### License 
See [LICENSE.md](./LICENSE.md) for details on this repo. This project is a fork of the [awesome-image-tools repo](https://github.com/underwatervideo/awesome-image-tools). The original [LICENSE](./LICENSE) from the upstream repository is retained in this fork.
