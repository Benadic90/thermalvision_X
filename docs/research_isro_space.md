# THERMAVISION-X: ISRO Missions & IR Image Colorization Research
## Physics-Guided Zero-Shot Infrared Colorization for ISRO's Bharatiya Antariksh Hackathon 2026
**Researched by**: Benad | June 2026

---

## Executive Summary

This document presents comprehensive research findings on ISRO's satellite missions, disaster management systems, agricultural monitoring programs, and space exploration initiatives -- identifying specific use cases where **THERMAVISION-X**, a Physics-Guided Zero-Shot Infrared Colorization system, would add transformative value. Our research covers **8 operational satellite programs**, **3 upcoming missions**, **5 disaster management application areas**, and **4 space exploration missions**, with specific attention to thermal infrared imaging capabilities that form the input domain for our colorization technology.

**Key Finding**: ISRO operates and plans multiple satellites with thermal infrared (TIR) imaging capabilities -- including INSAT-3D/3DR (TIR at 4km), Oceansat-3 (LWIR at 1km), GISAT-1 (MX-LWIR at 1.5km), and the upcoming TRISHNA mission (4-band TIR at 57m). These systems generate vast quantities of single-band and multi-band thermal imagery that require expert interpretation. THERMAVISION-X bridges this critical gap by converting thermal infrared images into physics-accurate natural color representations, making thermal data accessible to non-experts while preserving scientific integrity.

---

## Table of Contents

1. [ISRO's Earth Observation Satellites with Thermal Capabilities](#1-isros-earth-observation-satellites)
2. [ISRO's Disaster Management Support](#2-isros-disaster-management-support)
3. [ISRO's Agricultural Monitoring](#3-isros-agricultural-monitoring)
4. [ISRO's Space Exploration Missions](#4-isros-space-exploration-missions)
5. [Specific Use Cases for IR Colorization](#5-specific-use-cases-for-ir-colorization)
6. [Compelling Problem Statement for ISRO](#6-problem-statement-for-isro)
7. [Real-World Impact Scenarios](#7-real-world-impact-scenarios)
8. [India's Disaster Statistics](#8-indias-disaster-statistics)
9. [Data Availability & Access](#9-data-availability--access)
10. [References](#10-references)

---

## 1. ISRO's Earth Observation Satellites

### 1.1 Resourcesat-2 / Resourcesat-2A

| Parameter | Specification |
|-----------|--------------|
| **Full Name** | Resourcesat-2 (2011) / Resourcesat-2A (2016) |
| **Operator** | ISRO |
| **Orbit** | Sun-synchronous polar orbit, 817 km altitude |
| **Local Time** | 10:30 AM (descending node) |
| **Launch Vehicle** | PSLV-C16 (RS-2), PSLV-C36 (RS-2A) |
| **Mission Life** | 5+ years |
| **Launch Mass** | 1,235 kg (RS-2A) |

#### Sensor Specifications

| Sensor | Bands | Spectral Range | Resolution | Swath | Revisit |
|--------|-------|---------------|------------|-------|---------|
| **LISS-4** | 3 (B2, B3, B4) | 0.52-0.59, 0.62-0.68, 0.77-0.86 um | 5.8 m | 70 km | 5 days (with tilt) |
| **LISS-3** | 4 (B2, B3, B4, B5) | 0.52-0.59, 0.62-0.68, 0.77-0.86, 1.55-1.70 um | 23.5 m | 140 km | 24 days |
| **AWiFS** | 4 (B2, B3, B4, B5) | 0.52-0.59, 0.62-0.68, 0.77-0.86, 1.55-1.70 um | 56 m | 740 km | 5 days |

**Thermal Capabilities**: Resourcesat-2/2A does NOT have dedicated thermal infrared bands. However, the **SWIR band (1.55-1.70 um)** on LISS-3 and AWiFS provides thermal-adjacent information useful for temperature-related analysis. The NIR band (0.77-0.86 um) is used for vegetation stress monitoring via thermal proxies.

**Applications**: Crop discrimination, vegetation monitoring, water resource mapping, land use/land cover mapping, disaster management support, snow/glacier studies, coastal zone management.

**IR Colorization Use Case**: While Resourcesat lacks true thermal bands, colorization of SWIR-band imagery (1.55-1.70 um) can enhance water body detection, soil moisture mapping, and vegetation stress analysis. Multi-spectral fusion with thermal data from other ISRO satellites (INSAT, Oceansat) can create colorized composite products for agricultural advisory.

---

### 1.2 Cartosat-3

| Parameter | Specification |
|-----------|--------------|
| **Full Name** | Cartographic Satellite-3 |
| **Launch Date** | November 27, 2019 |
| **Operator** | ISRO |
| **Orbit** | Sun-synchronous, 509 km altitude, 97.5 deg inclination |
| **Local Time** | 9:30 AM |
| **Launch Vehicle** | PSLV-C47 |
| **Launch Mass** | 1,625 kg |
| **Power** | 2,000 W |
| **Mission Life** | 5 years |

#### Sensor Specifications

| Sensor | Bands | Spectral Range | Resolution | Swath |
|--------|-------|---------------|------------|-------|
| **PAN** | 1 | 0.45-0.90 um | **0.28 m** | 17 km |
| **Multispectral** | 4 (Blue, Green, Red, NIR) | 0.45-0.52, 0.52-0.59, 0.62-0.68, 0.77-0.86 um | **1.12 m** | 17 km |

**Key Capabilities**: High-agility platform with Control Moment Gyros (CMGs), stereo imaging, 4-day revisit, 11-bit radiometric resolution, X-band (960 Mbps) and Ka-band (2880 Mbps) data transmission.

**Applications**: Cadastral-level mapping, urban planning, precision agriculture, coastal land use, disaster assessment, infrastructure monitoring, 1:2500 scale thematic mapping.

**IR Colorization Use Case**: Cartosat-3's panchromatic band (0.45-0.90 um) captures some near-infrared information. Colorization can enhance change detection after disasters by creating natural-looking thermal-stress maps when fused with INSAT TIR data. Urban heat island analysis at 0.28m resolution (when combined with thermal downscaling) can identify building-level thermal anomalies.

---

### 1.3 INSAT-3D / INSAT-3DR / INSAT-3DS

| Parameter | Specification |
|-----------|--------------|
| **Full Name** | Indian National Satellite System - 3D Series |
| **Launch Dates** | INSAT-3D: July 26, 2013; INSAT-3DR: September 8, 2016; INSAT-3DS: Feb 2024 |
| **Operator** | ISRO / IMD |
| **Orbit** | Geostationary (36,000 km), 82 deg E |
| **Launch Mass** | 2,211 kg (INSAT-3DR) |
| **Power** | 1,700 W (solar array) |

#### Imager Specifications (6 Channels)

| Channel | Spectral Band | Wavelength | Spatial Resolution | Application |
|---------|--------------|------------|-------------------|-------------|
| 1 | Visible (VIS) | 0.55-0.75 um | **1 km** | Cloud imaging, snow cover |
| 2 | Shortwave IR (SWIR) | 1.55-1.70 um | **1 km** | Nighttime low cloud/fog detection |
| 3 | Midwave IR (MIR) | 3.80-4.00 um | **4 km** | Low cloud/fog at night, fire detection |
| 4 | Water Vapor (WV) | 6.50-7.10 um | **8 km** | Atmospheric moisture |
| 5 | **Thermal IR-1 (TIR-1)** | **10.3-11.3 um** | **4 km** | **SST, LST, cloud top temperature** |
| 6 | **Thermal IR-2 (TIR-2)** | **11.5-12.5 um** | **4 km** | **SST with improved accuracy** |

#### Sounder Specifications (19 Channels)
- **Shortwave IR**: 6 bands (3.74-4.57 um)
- **Midwave IR**: 5 bands (6.51-7.43 um)
- **Longwave IR**: 7 bands (11.03-14.71 um)
- **Visible**: 1 band (0.695 um)
- **Resolution**: 10 km x 10 km (all bands)

**Key Improvements in INSAT-3DR over INSAT-3D**:
- Middle Infrared band for **nighttime pictures of low clouds and fog**
- **Two Thermal Infrared bands** for SST estimation with better accuracy
- **Higher spatial resolution** in Visible and TIR bands
- Data Relay Transponder and Search & Rescue Transponder

**THIS IS ISRO'S PRIMARY THERMAL IMAGING ASSET.** The dual TIR bands (10.3-11.3 um and 11.5-12.5 um) are directly usable with THERMAVISION-X for colorization.

**IR Colorization Use Cases**:
1. **Nighttime weather visualization**: Colorize MIR (3.8 um) nighttime imagery to create intuitive fog/low-cloud maps for aviation and transportation safety
2. **SST visualization**: Convert dual-TIR sea surface temperature data into color-coded intuitive ocean temperature maps for fishery advisory (PFZ - Potential Fishing Zone)
3. **Disaster monitoring**: Colorize TIR imagery during cyclones to show thermal structure of storm systems for public communication
4. **Urban heat monitoring**: Colorize TIR-1/TIR-2 data to create accessible heat maps for city planners and public health officials

---

### 1.4 Oceansat-3 (EOS-06)

| Parameter | Specification |
|-----------|--------------|
| **Full Name** | OceanSat-3 / EOS-06 |
| **Launch Date** | November 26, 2022 |
| **Operator** | ISRO |
| **Orbit** | Sun-synchronous, 729 x 748 km, 98.34 deg |
| **Launch Mass** | 1,117 kg |

#### Sensor Specifications

| Sensor | Bands | Spectral Range | Resolution | Swath |
|--------|-------|---------------|------------|-------|
| **OCM** | 13 bands | 400-1010 nm (VNIR) | 360 m | 1,400 km |
| **LWIR (Thermal)** | **2 bands** | **~11 um and ~12 um** | **1,080 m** | - |
| **Scatterometer** | Ku-band | - | 50 km x 50 km | - |

**Key Thermal Capability**: Oceansat-3 is ISRO's first satellite with **dedicated long-wave infrared (LWIR) thermal channels** for Sea Surface Temperature (SST) measurement, using the split-window technique with two thermal bands around 11 and 12 um.

**Applications**: Ocean color monitoring, potential fishing zone (PFZ) forecasting, primary productivity estimation, sea surface temperature mapping, wind vector measurement for cyclone forecasting, numerical weather modeling.

**IR Colorization Use Cases**:
1. **Fishery visualization**: Colorize SST thermal imagery into intuitive temperature maps showing thermal fronts and eddies that attract fish -- directly supporting ISRO's operational PFZ forecasts
2. **Cyclone thermal structure**: Colorize LWIR data during cyclone events to visualize storm thermal signatures for coastal communities
3. **Ocean current visualization**: Create colorized thermal maps showing ocean current boundaries for maritime navigation

---

### 1.5 RISAT-1A (EOS-04)

| Parameter | Specification |
|-----------|--------------|
| **Full Name** | Radar Imaging Satellite-1A / Earth Observation Satellite-04 |
| **Launch Date** | February 14, 2022 |
| **Operator** | ISRO |
| **Orbit** | Sun-synchronous, 524.87 km, 97.5 deg, LTAN 6:00 |
| **Launch Mass** | 1,710 kg |
| **Power** | 2,280 W (solar array) |
| **Frequency** | C-band, 5.4 GHz (shifted from 5.35 GHz to avoid WLAN interference) |

#### Imaging Modes

| Mode | Resolution | Swath | Polarization |
|------|-----------|-------|-------------|
| HRS (High Resolution Spotlight) | **1 m** | 10 km | HH/VV/HH+HV/VV+VH |
| FRS-1 (Fine Resolution) | **3 m** | 25-30 km | Single/Dual Pol |
| FRS-2 | **9 m** | 30 km | Dual Pol |
| MRS (Medium Resolution) | **25 m** | 120-160 km | Single/Dual Pol |
| CRS (Coarse Resolution) | **50 m** | 223-240 km | Single/Dual Pol |

**Key Feature**: All-weather, day-and-night imaging using Synthetic Aperture Radar (SAR). While RISAT is not a thermal imager, it provides **critical complementary data** for thermal analysis -- particularly soil moisture estimation, flood inundation mapping, and crop monitoring that benefit from fusion with thermal infrared data.

**IR Colorization Use Case**: RISAT-1A SAR data combined with thermal imagery from INSAT-3D can be fused to create colorized disaster response maps. For example, SAR-derived flood extent overlaid on colorized thermal backgrounds provides comprehensive situational awareness for rescue operations.

---

### 1.6 GISAT-1 (EOS-03) -- GEO Imaging Satellite

| Parameter | Specification |
|-----------|--------------|
| **Full Name** | Geo Imaging Satellite-1 / EOS-03 |
| **Launch Date** | August 12, 2021 (failed to reach orbit) |
| **Follow-on** | GISAT-1A (EOS-05) -- planned Q1 2026 |
| **Operator** | ISRO |
| **Orbit** | Geostationary, 36,000 km, 83 deg E |
| **Launch Mass** | 2,268 kg |
| **Power** | 2,280 W |

#### Sensor Specifications

| Sensor | Channels | Resolution | Spectral Range |
|--------|----------|------------|---------------|
| **MX-VNIR** | 6 | **42 m** | 0.45-0.875 um |
| **HySI-VNIR** | 158 | **318 m** | 0.375-1.0 um |
| **HySI-SWIR** | 256 | **191 m** | 0.9-2.5 um |
| **MX-LWIR (Thermal)** | **6** | **1,180 m (1.18 km)** | **7.1-13.5 um** |

#### MX-LWIR Thermal Bands

| Channel | Wavelength Range | Application |
|---------|-----------------|-------------|
| CH1 | 7.1-7.6 um | Atmospheric correction |
| CH2 | 8.3-8.7 um | Surface emissivity |
| CH3 | 9.4-9.8 um | Ozone absorption |
| CH4 | 10.3-11.3 um | Split-window SST/LST |
| CH5 | 11.5-12.5 um | Split-window SST/LST |
| CH6 | 13.0-13.5 um | Cloud detection |

**Key Advantage**: GISAT provides **near real-time** imaging from geostationary orbit:
- Sector-wise image every **5 minutes**
- Full Indian landmass image every **30 minutes** at 42m resolution
- 1.18 km thermal imaging every 30 minutes

**THIS IS A CRITICAL PLATFORM FOR THERMAL COLORIZATION** -- it provides unprecedented temporal resolution for thermal monitoring.

**IR Colorization Use Cases**:
1. **Real-time thermal weather visualization**: Colorize 6-band thermal imagery every 30 minutes for TV broadcast and public weather communication
2. **Nowcasting support**: Create colorized thermal maps for short-range weather prediction (convective development, fog forecasting)
3. **Cyclone thermal structure**: Track cyclone thermal signatures in colorized format for real-time disaster communication
4. **Urban heat monitoring**: 30-minute revisit enables tracking of diurnal urban heat island dynamics in colorized format

---

### 1.7 TRISHNA Mission (Upcoming -- ISRO-CNES Joint)

| Parameter | Specification |
|-----------|--------------|
| **Full Name** | Thermal infraRed Imaging Satellite for High-resolution Natural resource Assessment |
| **Agencies** | ISRO + CNES (France) |
| **Status** | Approved, in development |
| **Launch Date** | **2026** (planned) |
| **Orbit** | Sun-synchronous, 761 km, LTDN 13:00 |
| **Altitude** | 761 km |
| **Revisit** | **Every 3 days** (3 passes per 8-day cycle) |
| **Launch Vehicle** | PSLV |
| **Mission Life** | 5-7 years |
| **Data Policy** | **Free and open** |

#### TIR Instrument (CNES-provided)

| Parameter | Specification |
|-----------|--------------|
| **Spectral Bands** | 4 thermal channels |
| **TIR-1** | 8.65 um (FWHM: 0.35 um) |
| **TIR-2** | 9.00 um (FWHM: 0.35 um) |
| **TIR-3** | 10.60 um (FWHM: 0.7 um) |
| **TIR-4** | 11.60 um (FWHM: 1.0 um) |
| **Spatial Resolution** | **57 m (land/coastal), 1 km (deep ocean)** |
| **Radiometric Accuracy** | 0.7 K at 300 K |
| **Precision (NEdT)** | **0.15 K at 300 K** |
| **Dynamic Range** | 250 K - 400 K |
| **Swath** | 1,026 km |
| **Scan Angle** | +/- 34 deg |

#### VSWIR Instrument (ISRO-provided)

| Band | Wavelength | Application |
|------|-----------|-------------|
| Blue | 485 nm | Low cloud detection |
| Green | 555 nm | Coastal, sediments, snow |
| Red | 670 nm | Vegetation (LAI, NDVI) |
| NIR | 860 nm | Vegetation monitoring |
| Cirrus | 1380 nm | Thin cirrus detection |
| SWIR | 1610 nm | Snow/cloud discrimination |

**THIS IS THE MOST IMPORTANT MISSION FOR THERMAVISION-X.** TRISHNA will provide:
- **57-meter resolution thermal imaging** -- the finest thermal resolution ISRO will have
- **4 thermal bands** enabling sophisticated temperature/emissivity separation
- **3-day global revisit** -- unprecedented temporal frequency
- **Free and open data policy** -- accessible for research and applications

**Scientific Themes**: Ecosystem water stress, coastal/inland water monitoring, urban heat islands, cryosphere, solid Earth, atmosphere.

**IR Colorization Use Cases**:
1. **Agricultural water stress maps**: Colorize thermal data to show crop water stress at field level (57m resolution) for farmer advisory
2. **Urban heat island visualization**: Create colorized thermal maps of Indian cities at unprecedented 57m resolution
3. **Disaster response**: Colorize rapid 3-day revisit thermal data for flood, drought, and fire monitoring
4. **Climate monitoring**: Generate colorized long-term thermal time series for climate change communication

---

### 1.8 NISAR (NASA-ISRO SAR Mission)

| Parameter | Specification |
|-----------|--------------|
| **Launch Date** | **July 30, 2025** (successful) |
| **Agencies** | NASA (JPL) + ISRO |
| **Orbit** | Sun-synchronous dawn-dusk, 747 km, 98.4 deg |
| **Repeat Cycle** | 12 days |
| **Launch Vehicle** | GSLV-F16 |
| **Launch Mass** | ~2,800 kg |
| **Mission Life** | 3 years (5 years consumables) |
| **Data Policy** | **Free and open** |

#### Dual-Frequency SAR

| Parameter | L-band (NASA) | S-band (ISRO) |
|-----------|--------------|---------------|
| **Frequency** | 1.257 GHz | 3.2 GHz |
| **Wavelength** | 24 cm | 10 cm |
| **Resolution** | 3-48 m (range), 7 m (azimuth) | 3-24 m (range), 7 m (azimuth) |
| **Swath** | >240 km | >240 km |
| **Polarization** | Quad Pol available | Single/Dual/Compact Pol |

**Note**: NISAR is a radar mission without thermal infrared bands. However, its all-weather capability complements thermal imaging for disaster response and agricultural monitoring.

---

## 2. ISRO's Disaster Management Support

### 2.1 Decision Support Centre (DSSC)

ISRO's Decision Support Centre operates at the National Remote Sensing Centre (NRSC) in Hyderabad, providing end-to-end disaster monitoring services using satellite data. The DSSC generates and disseminates:

- **Flood inundation maps** (within hours of satellite pass)
- **Landslide inventory maps**
- **Forest fire detection and monitoring**
- **Cyclone track prediction support**
- **Earthquake damage assessment**
- **Drought monitoring and assessment**

### 2.2 Bhuvan -- ISRO's Geospatial Platform

Bhuvan (https://bhuvan.nrsc.gov.in) is ISRO's flagship geoportal providing:
- Satellite imagery and thematic maps
- Disaster-specific services (flood, cyclone, landslide, fire)
- 2D/3D visualization of Indian terrain
- Near real-time disaster product dissemination

### 2.3 National Database for Emergency Management (NDEM)

NDEM integrates satellite-derived disaster products with ground information for:
- Multi-hazard disaster risk assessment
- Emergency response coordination
- Damage assessment and recovery planning

### 2.4 MOSDAC -- Meteorological & Oceanographic Data Archive

MOSDAC (https://www.mosdac.gov.in) archives and distributes:
- INSAT-3D/3DR imager and sounder data
- Oceansat series data
- Meteorological products (rainfall, SST, wind)
- **All TIR thermal imagery** from INSAT and Oceansat missions

### 2.5 Forest Fire Monitoring

ISRO has developed a **customized mobile application and dashboard** for forest fire reporting using geospatial technology. Key capabilities:
- Near real-time forest fire detection using INSAT-3D MIR channel (3.8 um)
- Fire hotspot mapping and dissemination via Bhuvan
- Fire spread prediction and risk assessment

### 2.6 Flood Monitoring

ISRO provides operational flood inundation mapping:
- Using RISAT-1A SAR data (all-weather, day/night)
- Using Resourcesat optical data (AWiFS, LISS-3, LISS-4)
- INSAT-3D thermal data for flood extent analysis
- Integration with meteorological forecasts

### 2.7 Urban Heat Island Monitoring

ISRO supports urban heat island studies using:
- INSAT-3D TIR data (4 km resolution) for city-scale analysis
- Landsat thermal data (100m) for detailed urban studies
- **Upcoming TRISHNA** (57m) for block-level thermal mapping
- **Upcoming GISAT-1A** (1.18km) for 30-minute diurnal monitoring

---

## 3. ISRO's Agricultural Monitoring

### 3.1 FASAL Program

| Parameter | Details |
|-----------|---------|
| **Full Name** | Forecasting Agricultural output using Space, Agrometeorology and Land based observations |
| **Implementing Agency** | Mahalanobis National Crop Forecast Centre (MNCFC), DAC&FW |
| **Satellite Inputs** | Resourcesat-2/2A, RISAT-1A, Sentinel-1A |
| **Coverage** | 20 states, 557 districts |
| **Crops Monitored** | 11 major crops (Paddy, Wheat, Jute, Cotton, Sugarcane, Soybean, Tur, Gram, Mustard, Lentil, Rabi Sorghum) |

#### FASAL Methodology

1. **Crop Mapping**: Uses multi-date SAR (RISAT-1A) for Rice/Jute; multi-date optical (Resourcesat AWiFS at 56m) for other crops
2. **Area Estimation**: Stratified random sampling with hierarchical classification
3. **Yield Estimation**: Weather-based agrometeorological models + remote sensing indices
4. **Forecast Output**: 16 forecasts per agricultural year at national/state/district levels

**IR Colorization Value**: FASAL outputs are primarily statistical. Colorized thermal imagery can make crop stress and soil moisture information directly visible and interpretable by farmers and extension workers.

### 3.2 Crop Health Monitoring via Thermal Infrared

Key thermal infrared applications in agriculture:
- **Crop water stress index (CWSI)**: Derived from canopy temperature via thermal imaging
- **Evapotranspiration mapping**: Using surface energy balance with TIR inputs
- **Soil moisture estimation**: Thermal inertia method using diurnal temperature difference
- **Pest/disease detection**: Thermal signatures of stressed vegetation (upcoming TRISHNA enables field-level detection at 57m)

### 3.3 YESTech -- Yield Estimation System using Technology

Under PMFBY (Pradhan Mantri Fasal Bima Yojana), ISRO supports:
- Gram Panchayat-level yield estimation for Paddy, Wheat, Soybean
- Crop Cutting Experiment (CCE) planning using satellite data
- Addressing yield and area discrepancies using EO data

---

## 4. ISRO's Space Exploration Missions

### 4.1 Chandrayaan-3 (Lunar Mission)

| Parameter | Specification |
|-----------|--------------|
| **Launch Date** | July 14, 2023 |
| **Landing Site** | 69.37 deg S, 32.35 deg E (Shiv Shakti Point) |
| **Operator** | ISRO |

#### Thermal Instrument: ChaSTE (Chandra's Surface Thermophysical Experiment)

| Parameter | Specification |
|-----------|--------------|
| **Type** | Thermal probe with 10 Platinum RTD sensors |
| **Measurement** | Regolith temperature (passive) + thermal conductivity (active) |
| **Depth** | Multi-point along probe length |
| **Key Finding** | Surface temperatures ~330 K (higher than LRO Diviner predictions of 273-300 K) |

**IR Colorization Use Case**: While ChaSTE provides point measurements, colorization of lunar thermal maps can enhance scientific visualization of surface thermophysical properties for research communication.

### 4.2 Aditya-L1 (Solar Observatory)

| Parameter | Specification |
|-----------|--------------|
| **Launch Date** | September 2, 2023 |
| **Orbit** | Halo orbit around Sun-Earth L1 point (1.5 million km from Earth) |
| **Mission Life** | 5+ years |

#### Key Instruments with Thermal Relevance

| Instrument | Capability |
|------------|-----------|
| **VELC** | Visible Emission Line Coronagraph (corona imaging 1.05-3.0 Rsun) |
| **SUIT** | Solar Ultraviolet Imaging Telescope (200-400 nm, 1.4 arcsec resolution) |
| **SoLEXS** | Solar Low Energy X-ray Spectrometer (1-30 keV) |
| **HEL1OS** | High Energy L1 Orbiting X-ray Spectrometer (hard X-rays) |
| **ASPEX** | Solar wind particle experiment (ions) |
| **PAPA** | Plasma Analyzer (electrons) |
| **Magnetometer** | In-situ magnetic field measurement |

**Thermal Management Challenge**: SUIT detector passively cooled to **-55 deg C** using cold finger connected to passive radiator. The thermal control system maintains CCD dark current below 4 e-/pix/s.

**IR Colorization Use Case**: Solar imagery in UV/X-ray can be "colorized" to represent different temperature regimes of the solar atmosphere for educational and research visualization.

### 4.3 Gaganyaan (Human Spaceflight Mission)

| Parameter | Specification |
|-----------|--------------|
| **Status** | In development |
| **Planned Altitude** | 400 km LEO |
| **Crew Capacity** | 2-3 astronauts |
| **Mission Duration** | Up to 7 days |
| **Launch Vehicle** | HLVM3 (Human-rated LVM3) |
| **Budget** | Rs 20,193 crore (revised, for 8 missions) |

#### Thermal Management Systems

| System | Description |
|--------|-------------|
| **Thermal Protection System (TPS)** | Ablative heat shield using carbon-phenolic and silica-phenolic composites |
| **Peak Re-entry Temperature** | Exceeds **3,000 deg C** |
| **ECLSS** | Environmental Control and Life Support System maintains cabin environment |
| **Service Module Thermal** | Passive radiators + active heaters for temperature control |
| **Double-walled Structure** | Thermally isolates crew module from external heat loads |

**IR Colorization Use Case**: Thermal imaging of the crew module and heat shield during re-entry can be colorized for real-time mission monitoring, engineering analysis, and public engagement.

---

## 5. Specific Use Cases for IR Colorization

### Use Case 1: Nighttime Earth Observation & Fog Detection
**Satellite**: INSAT-3D/3DR (MIR 3.8 um, TIR 10-12 um)
**Problem**: MIR nighttime imagery of fog and low clouds is grayscale and unintuitive for non-experts
**THERMAVISION-X Solution**: Colorize MIR/TIR nighttime imagery into natural-looking representations where fog appears as white/gray, clear areas as dark, and temperature gradients as color transitions
**Value**: Aviation safety (fog warnings for airports), road transport safety, public weather communication
**Users**: IMD, Airports Authority of India, NHAI, general public

### Use Case 2: Disaster Response -- Flood, Fire, Cyclone
**Satellites**: INSAT-3D/3DR (TIR), Oceansat-3 (LWIR), RISAT-1A (SAR)
**Problem**: Thermal imagery from multiple sources is difficult to rapidly interpret during emergency response
**THERMAVISION-X Solution**: Unified colorization pipeline converts multi-source thermal data into consistent, intuitive color maps showing flood extent, fire hotspots, or cyclone thermal structure
**Value**: Faster decision-making by disaster managers; accessible information for rescue teams; public awareness
**Users**: NDMA, SDMA, NDRF, State Disaster Management Authorities

### Use Case 3: Agricultural Water Stress Advisory
**Satellite**: TRISHNA (upcoming, 57m TIR), Oceansat-3 (SST)
**Problem**: Thermal-derived crop water stress indices are not directly interpretable by farmers
**THERMAVISION-X Solution**: Colorize thermal imagery into vegetation-health-style maps where green=healthy, yellow=stressed, red=severely stressed
**Value**: Farmer-friendly irrigation advisory; precision agriculture support; crop insurance documentation
**Users**: MNCFC (FASAL), State Agriculture Departments, Krishi Vigyan Kendras, farmers

### Use Case 4: Urban Heat Island Mitigation Planning
**Satellites**: INSAT-3D (TIR 4km), GISAT-1A (TIR 1.18km, 30-min), TRISHNA (TIR 57m)
**Problem**: Urban heat island data is too coarse or too technical for urban planners
**THERMAVISION-X Solution**: Colorize multi-resolution thermal data to show neighborhood-level heat patterns, identify cooling intervention priorities
**Value**: Evidence-based urban greening policies; building code improvements; public health heat warnings
**Users**: Urban local bodies, Smart Cities Mission, public health departments

### Use Case 5: Ocean State & Fishery Advisory
**Satellite**: Oceansat-3 (LWIR ~11-12 um)
**Problem**: SST thermal imagery requires expert interpretation for fishery applications
**THERMAVISION-X Solution**: Colorize SST data into intuitive ocean temperature maps showing thermal fronts, eddies, and upwelling zones that attract fish
**Value**: Enhanced PFZ (Potential Fishing Zone) forecasts; reduced search time for fishermen; improved catch rates
**Users**: INCOIS (Indian National Centre for Ocean Information Services), fishing communities

### Use Case 6: Multi-Spectral Image Fusion for Scene Understanding
**Satellites**: Resourcesat-2/2A (LISS-4/3/AWiFS) + INSAT-3D (TIR) + Oceansat-3 (LWIR)
**Problem**: Combining optical and thermal data requires complex processing for unified visualization
**THERMAVISION-X Solution**: Physics-guided colorization naturally fuses thermal information with optical scene understanding, generating RGB-like images from IR inputs
**Value**: Unified visualization products for analysts; reduced processing time; consistent multi-mission products
**Users**: NRSC, State Remote Sensing Centres, research institutions

---

## 6. Problem Statement for ISRO

### Primary Problem Statement

> **"ISRO generates vast quantities of single-band and multi-band thermal infrared imagery from INSAT-3D/3DR, Oceansat-3, and upcoming TRISHNA and GISAT-1A missions. These images are essential for disaster management, agricultural monitoring, urban planning, and oceanographic applications. However, thermal infrared imagery is inherently grayscale and requires specialized expertise to interpret. This creates a critical bottleneck: the stakeholders who most need this information -- disaster responders, farmers, urban planners, and policymakers -- cannot directly interpret thermal data, leading to delayed decisions, reduced impact, and underutilized satellite assets."**

### Secondary Problem

> **"Existing colorization approaches require paired training data (thermal + optical), which is scarce for ISRO's specific sensor configurations and Indian geographic conditions. Moreover, physics-blind deep learning methods produce colorizations that violate thermal physics principles, making them unsuitable for scientific applications."**

### THERMAVISION-X Solution

THERMAVISION-X addresses this gap by:
1. **Physics-guided approach**: Ensures colorized outputs respect thermal radiation physics
2. **Zero-shot capability**: Works without paired training data for ISRO-specific sensors
3. **Multi-source compatibility**: Handles INSAT TIR, Oceansat LWIR, TRISHNA 4-band TIR, and GISAT MX-LWIR
4. **Scientific validity**: Preserves quantitative thermal information in colorized output
5. **Accessibility**: Makes thermal data interpretable by non-expert stakeholders

---

## 7. Real-World Impact Scenarios

### Scenario 1: Kerala Monsoon Floods -- Rapid Response Visualization

**Context**: During the 2024 monsoon, Kerala experienced severe flooding affecting 800+ villages across multiple districts. ISRO's DSSC generated flood inundation maps within hours, but the thermal imagery showing waterlogged areas remained in grayscale formats accessible only to remote sensing experts.

**THERMAVISION-X Impact**:
- Colorizes INSAT-3D TIR imagery showing flood thermal signatures within minutes of data acquisition
- Creates intuitive maps where deep floodwater=blue, shallow=submerged vegetation=green-brown, dry land=natural color
- Enables NDRF commanders to prioritize rescue operations without requiring remote sensing training
- Allows district collectors to communicate flood extent to local communities using visual maps
- Reduces interpretation time from hours to seconds

**Metrics**: 11,790 km2 flooded in Gujarat 2024 floods; 8+ million people exposed; interpretation bottleneck affects all events.

### Scenario 2: Telangana Cotton Belt -- Precision Irrigation Advisory

**Context**: Telangana's cotton farmers (32 districts under FASAL) face water scarcity and rely on groundwater irrigation. Current FASAL outputs are statistical forecasts, not actionable field-level guidance. TRISHNA's upcoming 57m thermal data will provide unprecedented field-level water stress information.

**THERMAVISION-X Impact**:
- Colorizes TRISHNA thermal data to show crop water stress at individual field level
- Generates farmer-friendly maps: dark green=healthy, light green=monitor, yellow=irrigate soon, red=critical
- Integrates with FASALSoft for operational crop advisory at MNCFC
- Enables gram panchayat-level irrigation scheduling through visual interpretation
- Supports PMFBY insurance claims with visual evidence of crop stress

**Metrics**: FASAL covers 557 districts, 11 crops, 20 states; precision irrigation can improve water use efficiency by 25-40%.

### Scenario 3: Delhi NCR Urban Heat Island -- Policy Intervention

**Context**: Delhi NCR experiences severe urban heat island effects with temperature differentials of 4-8 deg C between urban core and peripheral areas. Current INSAT-3D TIR data at 4km resolution is too coarse for neighborhood-level planning, but GISAT-1A (1.18km, 30-min) and TRISHNA (57m) will enable detailed analysis.

**THERMAVISION-X Impact**:
- Colorizes high-frequency GISAT-1A thermal data to track diurnal heat island evolution
- Creates visual heat maps for urban planners showing hottest neighborhoods requiring greening intervention
- Supports Smart Cities Mission with before/after visualization of cooling interventions
- Enables public health departments to issue targeted heat warnings for vulnerable areas
- Provides evidence base for building code revisions (cool roofing mandates, green space requirements)

**Metrics**: India's urban population will reach 600 million by 2030; heat waves caused 733 deaths in India in 2024; urban heat island mitigation can reduce cooling energy demand by 20-40%.

---

## 8. India's Disaster Statistics

### 8.1 Key Disaster Statistics for India (2024)

| Statistic | Value | Source |
|-----------|-------|--------|
| **Disaster events in India (2024)** | 15 major events | EM-DAT/IRDR |
| **Disaster-related deaths in India (2024)** | 1,507 fatalities | EM-DAT 2024 |
| **India rank in Asia disaster frequency** | 5th (after US, China, Indonesia, Philippines) | EM-DAT 2024 |
| **Population exposed to flooding (2024)** | 8+ million (Gujarat event alone) | IMD/NDMI |
| **Crop area damaged by floods (2024)** | 3,806 km2 total across events | Satellite assessment |
| **Houses destroyed by floods (2024)** | 20,137 houses | NDMI |
| **Road network damaged (2024)** | 5,081 km | State reports |
| **Heat wave deaths in India (2024)** | 733 reported | EM-DAT |
| **Flood-prone area in India** | 49.815 million hectares | NDMA |
| **Average annual flooded area** | 7.17 million hectares | NDMA |

### 8.2 Global Context (2024)

| Metric | 2024 Value | 30-Year Average (1994-2023) | Trend |
|--------|-----------|---------------------------|-------|
| Global disaster occurrences | 393 | 332 | Increasing |
| Global disaster deaths | 16,753 | 65,566 | Decreasing |
| Global economic losses | USD 241.95 billion | USD ~100 billion | Increasing |
| Asia disaster events | 148 | 132 | Increasing |
| Asia economic losses | USD 31.9 billion | USD 55.8 billion | Decreasing* |

*Note: Decreasing trend may reflect incomplete 2024 data.

### 8.3 Thermal Imaging Relevance to Indian Disasters

| Disaster Type | Annual Impact | Thermal Imaging Role |
|--------------|--------------|---------------------|
| **Floods** | 7.17 Mha average inundation; 128 events globally in 2024 | Flood extent mapping (water thermal signature), rescue prioritization |
| **Heat Waves** | 733+ deaths in India 2024; 50+ deg C temperatures | Urban heat island detection, vulnerable population identification |
| **Cyclones** | 4-6 major cyclones/year in Indian Ocean | SST measurement for intensity prediction, thermal structure analysis |
| **Forest Fires** | Thousands of fire points detected annually | Fire hotspot detection (MIR 3.8 um), fire spread monitoring |
| **Droughts** | 11 events in Asia 2024 | Evapotranspiration mapping, soil moisture estimation via thermal inertia |
| **Urban Heat** | 4-8 deg C UHI intensity in major cities | Continuous thermal monitoring, mitigation planning |

### 8.4 Economic Impact

- Floods caused **64.8%** of global disaster economic losses in 2024
- India: Crop damage of 1,978 km2 in Telangana-AP floods; 1,828 km2 in Northern India floods (2024)
- Economic losses from disasters in Asia: USD 31.9 billion (2024)
- India's average annual flood damage: **Rs 1,800 crore** (NDMA estimate)

---

## 9. Data Availability & Access

### 9.1 ISRO Data Portals

| Portal | URL | Data Available | Access |
|--------|-----|---------------|--------|
| **Bhuvan** | bhuvan.nrsc.gov.in | All ISRO satellite imagery, disaster products | Free registration |
| **NDEM** | ndem.nrsc.gov.in | Disaster-specific products | Government agencies |
| **MOSDAC** | www.mosdac.gov.in | INSAT, Oceansat, meteorological data | Free |
| **Bhoonidhi** | bhoonidhi.nrsc.gov.in | Earth observation data archive | Free/paid |
| **PRADAN** | (via ISSDC) | Aditya-L1 data | Free |
| **VEDAS** | vedas.sac.gov.in | Value-added data products | Research access |

### 9.2 Data Availability by Mission

| Mission | Thermal Data Available | Resolution | Access Level |
|---------|----------------------|------------|-------------|
| **INSAT-3D/3DR/3DS** | TIR-1, TIR-2, MIR, WV | 4-8 km | Free via MOSDAC |
| **Oceansat-3** | LWIR (2 bands) | 1,080 m | Free via MOSDAC |
| **RISAT-1A** | SAR (C-band, not thermal) | 1-50 m | Bhoonidhi |
| **Resourcesat-2/2A** | SWIR (1 thermal-adjacent band) | 23.5-56 m | Bhoonidhi |
| **TRISHNA (2026)** | 4-band TIR | **57 m** | **Free and open** |
| **GISAT-1A (2026)** | 6-band LWIR | **1.18 km** | Government access |
| **NISAR** | L-band + S-band SAR | 3-48 m | **Free and open** |

### 9.3 International Data (Complementary)

| Source | Thermal Bands | Resolution | Use for THERMAVISION-X |
|--------|--------------|------------|----------------------|
| **Landsat 8/9 TIRS** | 2 TIR bands (10.9, 12.0 um) | 100 m (resampled to 30m) | Training data, validation |
| **ECOSTRESS** | 5 TIR bands | 38 x 57 m | High-resolution validation |
| **Sentinel-3 SLSTR** | 2 TIR bands | 1 km | SST comparison |
| **MODIS** | 2 TIR bands | 1 km | Large-area validation |
| **ASTER** | 5 TIR bands | 90 m | Temperature/emissivity validation |

---

## 10. How THERMAVISION-X Adds Value to Each Mission

### For INSAT-3D/3DR
- Colorizes 4km TIR data for weather broadcast and public communication
- Creates intuitive fog/low-cloud nighttime maps for aviation safety
- Enhances cyclone thermal structure visualization for disaster communication

### For Oceansat-3
- Colorizes 1km LWIR SST data into intuitive ocean temperature maps
- Enhances PFZ (Potential Fishing Zone) forecast products for fishermen
- Creates colorized ocean thermal front maps for maritime safety

### For TRISHNA (Upcoming)
- Provides colorization pipeline ready at launch for 57m 4-band TIR data
- Enables field-level agricultural advisory via colorized water stress maps
- Creates urban heat maps at unprecedented 57m resolution for city planning

### For GISAT-1A (Upcoming)
- Colorizes 30-minute revisit thermal data for real-time weather communication
- Enables diurnal urban heat island tracking with intuitive visualizations
- Supports nowcasting applications with colorized rapid-refresh thermal maps

### For Disaster Management (DSSC)
- Accelerates thermal data interpretation from hours to seconds
- Creates unified color maps from multi-source thermal data during emergencies
- Makes satellite-derived thermal information accessible to non-expert responders

### For FASAL (Agricultural Monitoring)
- Translates thermal-derived crop stress indices into farmer-friendly visual maps
- Supports precision irrigation advisory at field level
- Provides visual evidence for crop insurance claims under PMFBY

---

## 11. Technology Roadmap for Hackathon

### Phase 1: Proof-of-Concept (Hackathon)
- Demonstrate physics-guided IR colorization on INSAT-3D TIR data
- Show zero-shot capability on unseen thermal imagery
- Present 3 real-world use case demonstrations

### Phase 2: Sensor-Specific Adaptation
- Train/fine-tune for TRISHNA's 4-band TIR configuration
- Adapt for GISAT-1A's 6-band MX-LWIR
- Integrate with Oceansat-3's 2-band LWIR

### Phase 3: Operational Integration
- Deploy on ISRO's Bhuvan/NDEM portals
- Integrate with DSSC operational pipeline
- Connect to FASAL advisory system

---

## 12. Competitive Advantage

### Why THERMAVISION-X Wins

| Feature | Existing Solutions | THERMAVISION-X |
|---------|-------------------|----------------|
| **Training Data** | Requires paired IR+RGB datasets (scarce for ISRO sensors) | Zero-shot: no paired training needed |
| **Physics Consistency** | Deep learning produces "hallucinated" colors | Physics-guided: respects thermal radiation laws |
| **Sensor Adaptability** | Retraining needed for each new sensor | Works across INSAT, Oceansat, TRISHNA, GISAT |
| **Scientific Validity** | Colors are artistic, not physically meaningful | Colors encode physical temperature information |
| **Nighttime Capability** | Limited (needs visible reference) | Native thermal-to-color translation |
| **Multi-band Support** | Typically single-band | Handles 2-band, 4-band, 6-band TIR configurations |

---

## 13. References

1. ISRO Official Website: www.isro.gov.in
2. Resourcesat-2A System Characterization Report, USGS (2025)
3. Resourcesat-2 Handbook, ISRO/NRSC
4. Cartosat-3 Brochure, NRSC (January 2021)
5. INSAT-3DR Official Page, ISRO
6. MOSDAC INSAT-3DR Payloads Documentation
7. INSAT-3D/3DR/3DS Imager Specifications, Vayumandal Journal
8. Oceansat-3 Specifications, Gunter's Space Page
9. EOS-04 (RISAT-1A) Handbook, Bhoonidhi/NRSC
10. RISAT-1 Mission Summary, eoPortal
11. Chandrayaan-3 Details, ISRO
12. ChaSTE Observations, LPSC 2025
13. Aditya-L1 Official Page, ISRO
14. SUIT Instrument Paper, arXiv (January 2025)
15. eoPortal Aditya-L1 Mission Summary
16. Gaganyaan Official Page, ISRO
17. TRISHNA Mission Page, ISRO
18. TRISHNA eoPortal Summary (April 2024)
19. CNES TRISHNA Project Page
20. NISAR Mission Page, ISRO/NASA
21. NISAR eoPortal Summary (September 2025)
22. FASAL Program, PIB Press Release (December 2025)
23. Remote Sensing for Crop Forecasting, FAO (2014)
24. ISRO Disaster Management Capabilities, ResearchGate
25. 2024 India Flood Assessment, AGU Earth's Future
26. 2024 Global Natural Disaster Assessment Report, IRDR
27. 2024 Disasters in Numbers, CRED/EM-DAT
28. Urban Heat Island Remote Sensing Review, Atmosphere Journal (2025)
29. GISAT-1 Specifications, ISRO PIB Release
30. GISAT-1 Technical Details, IOCCG

---

## Appendix A: Spectral Band Summary for ISRO Thermal Instruments

| Mission | Instrument | Band | Wavelength (um) | Resolution | Type |
|---------|-----------|------|----------------|------------|------|
| INSAT-3D/3DR | Imager | TIR-1 | 10.3-11.3 | 4 km | Thermal IR |
| INSAT-3D/3DR | Imager | TIR-2 | 11.5-12.5 | 4 km | Thermal IR |
| INSAT-3D/3DR | Imager | MIR | 3.8-4.0 | 4 km | Mid IR |
| INSAT-3D/3DR | Imager | WV | 6.5-7.1 | 8 km | Water Vapor |
| INSAT-3D/3DR | Sounder | LWIR | 11.0-14.7 | 10 km | Thermal IR |
| Oceansat-3 | LWIR | Band 1 | ~11.0 | 1,080 m | Thermal IR |
| Oceansat-3 | LWIR | Band 2 | ~12.0 | 1,080 m | Thermal IR |
| GISAT-1A | MX-LWIR | CH1 | 7.1-7.6 | 1,180 m | Thermal IR |
| GISAT-1A | MX-LWIR | CH2 | 8.3-8.7 | 1,180 m | Thermal IR |
| GISAT-1A | MX-LWIR | CH3 | 9.4-9.8 | 1,180 m | Thermal IR |
| GISAT-1A | MX-LWIR | CH4 | 10.3-11.3 | 1,180 m | Thermal IR |
| GISAT-1A | MX-LWIR | CH5 | 11.5-12.5 | 1,180 m | Thermal IR |
| GISAT-1A | MX-LWIR | CH6 | 13.0-13.5 | 1,180 m | Thermal IR |
| TRISHNA | TIR | TIR-1 | 8.65 (0.35 FWHM) | 57 m | Thermal IR |
| TRISHNA | TIR | TIR-2 | 9.00 (0.35 FWHM) | 57 m | Thermal IR |
| TRISHNA | TIR | TIR-3 | 10.60 (0.7 FWHM) | 57 m | Thermal IR |
| TRISHNA | TIR | TIR-4 | 11.60 (1.0 FWHM) | 57 m | Thermal IR |

---

## Appendix B: Mission Status Summary

| Mission | Status | Launch Year | Primary Thermal Bands |
|---------|--------|-------------|----------------------|
| Resourcesat-2 | Operational | 2011 | SWIR (1 thermal-adjacent) |
| Resourcesat-2A | Operational | 2016 | SWIR (1 thermal-adjacent) |
| INSAT-3D | Operational | 2013 | 2 TIR + 1 MIR + 1 WV |
| INSAT-3DR | Operational | 2016 | 2 TIR + 1 MIR + 1 WV |
| INSAT-3DS | Operational | 2024 | 2 TIR + 1 MIR + 1 WV |
| Cartosat-3 | Operational | 2019 | NIR only |
| Oceansat-3 | Operational | 2022 | 2 LWIR |
| RISAT-1A (EOS-04) | Operational | 2022 | SAR (not thermal) |
| Chandrayaan-3 | Mission Complete | 2023 | ChaSTE (in-situ thermal probe) |
| Aditya-L1 | Operational | 2023 | UV/X-ray (not TIR) |
| NISAR | Operational | 2025 | SAR (not thermal) |
| Gaganyaan-1 | Planned | 2026 | Thermal Protection System |
| Gaganyaan-2 | Planned | 2026-2027 | Thermal Protection System |
| GISAT-1A (EOS-05) | Planned | Q1 2026 | 6-band MX-LWIR |
| TRISHNA | Planned | 2026 | **4-band TIR at 57m** |
| Resourcesat-3 | Planned | 2026-2027 | Enhanced SWIR + VNIR |

---

*Document prepared for THERMAVISION-X -- Physics-Guided Zero-Shot Infrared Colorization System*
*ISRO Bharatiya Antariksh Hackathon 2026*
*Last updated: July 2025*
