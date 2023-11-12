# NAWEA WindTech 2023 WOMBAT Tutorial

## What You'll Need

- [COREWIND Floating Wind Description](https://corewind.eu/wp-content/uploads/files/publications/COREWIND-D6.1-General-frame-of-the-analysis-and-description-of-the-new-FOW-assessment-app.pdf)
- [COREWIND Floating Wind O&M Strategies Assessment](https://corewind.eu/wp-content/uploads/files/publications/COREWIND-D4.2-Floating-Wind-O-and-M-Strategies-Assessment.pdf)
- Code editor (VSCode is what I use and therefore recommend)
- Working python environment (Miniconda is my preference because it's lightweight)
  - [Anaconda](https://docs.anaconda.com/free/anaconda/install)
  - [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)
- Basic git proficiency (we just need to clone a project and change branches)
- Basic terminal proficiency (or Windows alternative)
- Basic Python proficiency (WOMBAT requires very little code, but you still need to use Python)

## Slides and Data

Just note that the spoken commentary is not included, but the materials to drive that
content are all included in the slides with links to the appropriate documentation pages
and the relevant screenshots so that participants can track down the required data in
the COREWIND publications more easily.

- The slides can be found [here](https://www.nrel.gov/docs/fy24osti/87913.pdf)
- The accompanying example notebook is
  [here](https://github.com/WISDEM/WOMBAT/blob/main/examples/NAWEA_interactive_walkthrough.ipynb).
- Example dataset for the Morro Bay, California, USA 9D layout with connected
  substations is [here](https://github.com/WISDEM/WOMBAT/blob/main/library/corewind/).

## Pre-Workshop Setup

1. Create a new Python environment (conda instructions)

   ```bash
   conda create -n wombat_workshop python=3.10
   conda config --set pip_interop_enabled true
   ```

2. Download WOMBAT from GitHub

   ```bash
   git clone https://github.com/WISDEM/WOMBAT.git
   ```

3. Install WOMBAT as an editable package

   ```bash
   conda activate wombat_workshop  # or what you called in the first step
   cd wombat/
   pip install -e .
   ```

4. Ensure that it all worked (assuming no error messages at any of the prior stages)

   ```bash
   python

   >>> import wombat
   >>> wombat.__version__
   ```
