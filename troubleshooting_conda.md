# Troubleshooting bites

### Installing Packages of Requirements.txt

In case you have issues at time to try installing packages by 
"conda install --file requirements.txt" command, try this:

``` conda config --append channels conda-forge ``` 

and do it again.

### TA-Lib Issues

the best way to install `TA-Lib` in conda environment is using the following command:

``` conda install -c conda-forge ta-lib ```




