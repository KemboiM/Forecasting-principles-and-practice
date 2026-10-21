# -----------------------------------------------------------------------------
# R/setup_python.R  -- run ONCE (from the project root in RStudio).
# Creates a Python virtual environment called "fpp" with every package
# -----------------------------------------------------------------------------
if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)

# If you have no Python yet, this installs a private one for reticulate:
if (is.null(tryCatch(py_discover_config()$python, error = function(e) NULL))) {
  install_python(version = "3.11")
}
if (!"fpp" %in% virtualenv_list()) virtualenv_create("fpp")
virtualenv_install("fpp", packages = readLines("requirements.txt")[!grepl("^#|^$", readLines("requirements.txt"))])
message("Done. Restart R (Session > Restart R); chapters will now use the 'fpp' virtualenv.")
