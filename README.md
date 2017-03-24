# Radiation_Calculator

This radiation calculator is being used for SpinLab for calculating the radiation that comes from a decaying isotope.

If a file is being used as the input, then the following fomat must be used.

Name of file must be in the fomat: z.a.isomer.~~.txt where z is the number of protons, and a is the number of nucleons of the isotope being described. In the file, there must be 2 columns seperated by tabs, where the first column is the date the isotope is added and the second is the amount. This will become a real unit later. The date can in the following formats:
month.day.year, month.day.year.hour, month.day.year.hour.minute, month.day.year.hour.minute.second. For example, 6.21.2016.14.23.42 would be June 21, 2016 at 2:23 and 42 seconds PM. An example for Ra-225 could be:

filename: 88.225.0.txt

6.21.2016 1

7.24.2016 2

8.02.2016 0.5
