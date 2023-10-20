# NAWEA WindTech 2023 WOMBAT Tutorial

## What You'll Need

- [COREWIND Floating Wind Description](https://corewind.eu/wp-content/uploads/files/publications/COREWIND-D6.1-General-frame-of-the-analysis-and-description-of-the-new-FOW-assessment-app.pdf) 
- [COREWIND Floating Wind O&M Strategies Assessment](https://corewind.eu/wp-content/uploads/files/publications/COREWIND-D4.2-Floating-Wind-O-and-M-Strategies-Assessment.pdf)
- Code editor (VSCode is what I use and therefore recommend)
- Working python environment (Miniconda is my preference because it's lightweight)
  - Anaconda: https://docs.anaconda.com/free/anaconda/install
  - Miniconda: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html
- Basic git proficiency (we just need to clone a project and change branches)
- Basic terminal proficiency (or Windows alternative)
- Basic Python proficiency (WOMBAT doesn't requires very little code, but you still need to use Python)

## Pre-Workshop Setup

1. Create a new Python environment (conda instructtions)

   ```
   conda create -n wombat_workshop python=3.10
   conda config --set pip_interop_enabled true
   ```

2. Download WOMBAT from GitHub

   ```
   git clone git clone https://github.com/WISDEM/WOMBAT.git
   ```

3. Install WOMBAT as an editable package

   ```
   conda activate wombat_workshop  # or what you called in the first step
   cd wombat/
   pip install -e .
   ```

4. Ensure that it all worked (assuming no error messages at any of the prior stages)

   ```
   python

   >>> import wombat
   >>> wombat.__version__
   ```
