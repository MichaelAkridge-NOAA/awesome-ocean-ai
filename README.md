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

## Starter Tools
- https://colab.research.google.com   | Free GPU Access

## Image Annotation
| Name | Free | Open Source | AI Assisted Annotation | Notes |
|---------------------------------------------------|------|-------------|------------------------|-----------------------------------------------------------------------------------------------|
| [Label Studio](https://github.com/HumanSignal/label-studio) | ✅| ✅ | ✅ | Multi-type data labeling and annotation tool with standardized output format. |
| [CVAT](https://github.com/opencv/cvat)            | ✅    | ✅  | ✅ | Open-source tool for video and image annotation. |
| [TagLab](https://github.com/cnr-isti-vclab/TagLab/) | ✅    | ✅  | ✅ |  |
| [VIAME](http://www.viametoolkit.org/)             | ✅    | ✅  | ✅ | See [documentation](https://viame.readthedocs.io/en/latest/section_links/documentation_overview.html). |
| [Tator](https://www.tator.io/)                    | ✅    | ✅  | | See [Tutorials](https://www.tator.io/tutorials) and [Source Code](https://github.com/cvisionai/Tator). |
| [VoTT (Archived)](https://github.com/Microsoft/VoTT) | ✅ | ✅ | | Officially archived, last update in 2020. |
| [BIIGLE](https://www.biigle.de/)                  | ✅    | ✅  | | Check [GitHub](https://github.com/biigle/). |
| [labelme](https://github.com/wkentaro/labelme)    | ✅    | ✅  | | Basic annotation tool for labeling images. |
| [Sebastes](https://repository.library.noaa.gov/view/noaa/11999/noaa_11999_DS1.pdf) | ✅ | ✅ | | Used for fish detection. |
| [Squidle+](https://bitbucket.org/ariell/marinedb) | ✅ | ✅ | | Marine biology database and annotation tool. |
| [FishID](https://globalwetlandsproject.org/tools-2/fishid/) | ✅ | 🔜 | | Email to request access: fishidglow@outlook.com. |
| [Coral Point Count with Excel extensions](https://cnso.nova.edu/cpce/index.html) | ✅ | ❌ | | Popular for coral reef studies. |
| [BenthoBox](https://benthobox.com)                | ✅ | ❌ | | Benthic imagery analysis tool. |
| [CoralNet](https://coralnet.ucsd.edu/)            | ✅ | ❌ | ✅ | Great for coral species identification. |
| [Supervisely](https://supervise.ly/)              | ❌ | ❌ | ✅ | Paid service, widely used for AI-based labeling. |
| [Deep Sea Spy](https://www.deepseaspy.com)        | ❌ | ❌ | | Online annotation tool. |
| [Labelbox](https://labelbox.com/)                 | ❌ | ❌ | ✅ | Leading platform for AI-assisted labeling. |
| [OFOP](http://www.ofop-by-sams.eu/)               | ❌ | ❌ | | Oceanographic annotation platform. |
| [RectLabel](https://rectlabel.com/)               | ❌ | ❌ | | Paid video/image annotation tool for macOS. |
| [SeaGIS](https://www.seagis.com.au/)              | ❌ | ❌ | | Includes [EventMeasure](https://www.seagis.com.au/event.html) and [TransectMeasure](https://www.seagis.com.au/transect.html). |

---

## Video Annotation
| Name | Free | Open Source | Notes |
| -- | -- | -- | -- |
| [Label Studio](https://github.com/HumanSignal/label-studio) | ✅ | ✅ | Multi-type data labeling and annotation tool. |
| [CVAT](https://github.com/opencv/cvat) | ✅ | ✅ | Extensive video annotation support. |
| [BIIGLE](https://www.biigle.de/) | ✅ | ✅ | See [GitHub](https://github.com/biigle/). |
| [MBARI Media Management (M3/VARS)](https://mbari-media-management.github.io/) | ✅ | ✅ | Specialized tool for deep-sea video management. |
| [Scalabel](https://www.scalabel.ai/) | ✅ | ✅ | Modern video and image annotation platform. |
| [Tator](https://www.tator.io/) | ✅ | ✅ | See [Tutorials](https://www.tator.io/tutorials). |
| [VARS](https://hohonuuli.github.io/vars/) | ✅ | ✅ | Annotation and reference system for underwater video. |
| [video-annotation-tool](https://github.com/video-annotation-project/video-annotation-tool) | ✅ | ✅ | Lightweight, open-source tool. |
| [VIAME](http://www.viametoolkit.org/) | ✅ | ✅ | [Docs](https://viame.readthedocs.io/en/latest/section_links/documentation_overview.html). |
| [FishID](https://globalwetlandsproject.org/tools-2/fishid/) | ✅ | 🔜 | Email fishidglow@outlook.com for access. |
| [Digital Fishers](https://www.oceannetworks.ca/learning/get-involved/citizen-science/digital-fishers) | ❌ | ❌ | Citizen science platform for video classification. |
| [SeaTube](http://dmas.uvic.ca/SeaTube) | ✅ | ❌ | Tool for underwater video annotation. |
| [ADELIE](https://www.flotteoceanographique.fr/La-Flotte/Logiciels-embarques/ADELIE) | ✅ | ❌ | Marine research annotation tool. |

---

## AI / ML Backends
| Name | Link | Notes |
|------|------|-------|
| Nuclio | [Nuclio GitHub](https://github.com/nuclio/nuclio) | Serverless AI/ML platform for scalable computing. |
| Label Studio ML Backend | [Label Studio ML](https://github.com/HumanSignal/label-studio-ml-backend) | AI backends for Label Studio. |
| Tator ML Backend | [Tator GitHub](https://github.com/cvisionai/Tator-ML) | Backend system for AI-driven video annotation. |

---

## Models
| Name  | Link  | Description |
|-------|-------|-------------|
| FathomNet | [Hugging Face](https://huggingface.co/FathomNet) / [GitHub](https://github.com/fathomnet/models) | Models for marine species detection and analysis. |
| Fish Detector  | [YOLOv8n Grayscale Fish Detector](https://huggingface.co/akridge/yolo8-fish-detector-grayscale) | Grayscale YOLOv8 model trained to detect fish in marine imagery. |
| VIAME Model Zoo | [VIAME Model Zoo](https://github.com/VIAME/VIAME/wiki/Model-Zoo-and-Add-Ons) | Collection of models and add-ons for video and image-based marine analysis. |
| Placeholder | | |

---

## Identification Resources
- [Encyclopedia of Life](https://eol.org/)
- [MBARI Deep-Sea Guide](http://dsg.mbari.org)
- [Tree of Life](http://tolweb.org)
- [World Register of Marine Species (WoRMS)](http://www.marinespecies.org/)

---

## Training Data Sets
- [Hugging Face](https://huggingface.co/)
- [Roboflow](https://roboflow.com/)
- [Atlantis Dataset](https://github.com/smhassanerfani/atlantis)
- [FathomNet](http://fathomnet.org)
- [Fishnet.AI](https://www.fishnet.ai/)
- [FishIDLowViz](https://github.com/slopezmarcano/dataset-fish-detection-low-visibility)
- [FishIDLuderick](https://github.com/globalwetlands/luderick-seagrass)
- [DeepFish](https://alzayats.github.io/DeepFish/)
- [VIAME](https://viame.kitware.com)

---

## Miscellaneous
| Name | Free | Open Source | Notes |
| -- | -- | -- | -- |
| [3D Metrics](https://3d-metrics.com/) | ❌ | ❌ | 3D analysis for marine imagery. |
| [Cthulhu](https://github.com/mbari-media-management/cthulhu) | ✅ | ✅ | Video player with remote control API and localization support. |
| [iNaturalist](https://www.inaturalist.org/) | ✅ | ❌ | Biodiversity identification resource. |
| [Matisse](https://www.eso.org/sci/facilities/develop/instruments/matisse.html) | ❌ | ❌ | Tool for spectral analysis. |
| [OpenCV](https://opencv.org/) | ✅ | ✅ | Computer vision library for image/video processing. |
| [Sharktopoda](https://github.com/mbari-org/Sharktopoda) | ✅ | ✅ | Video player with remote control and bounding box support. |
| [FishIDTrackers](https://github.com/slopezmarcano/automated-fish-tracking) | ✅ | ✅ | Fish detection and tracking pipelines with demo datasets. |
| [Dive](https://github.com/Kitware/dive) | ✅ | ✅ | Annotation tool for 2D and 3D video data. |
