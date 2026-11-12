# FAQ

**Does ScaleCov fit a model?**
No. It works on residuals of any model. Residuals must be out of sample: produced by a model that did not see the pixel's unit, for example out-of-fold predictions.

**My units are polygons of different sizes. Is that a problem?**
No. Unit means are area weighted, every calibration unit has equal weight in the pixel quantile, and variances use each unit's own pixel geometry.

**Why does `variogram` under-cover even on my simulated data without shared errors?**
If the correlation range is comparable to the unit size, removing unit means hides part of the correlation. See [Recommended settings](settings.md).

**Can I use ScaleCov for non-agricultural maps?**
Yes. Nothing is specific to crops: any map of a continuous variable with nested units works, for example forest biomass by stand and district, or soil carbon by field and farm.
