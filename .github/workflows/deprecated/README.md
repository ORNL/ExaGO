This folder contains deprecated workflow configurations, such as
mirroring to other computing facilities. We are keeping these around
as a starting point if similar workflows need to be setup in the future,
but they will require some tinkering.


## `ornl_ascent_mirror.yaml`

This pushes to a GitLab at ORNL and runs CI/CD on Ascent. This also can re-build
modules for testing newer versions of ExaGO and it's dependencies without needing
to monitor builds by hand.

## `pnnl_mirror.yaml`

Similar to `ornl_ascent_mirror.yaml`, this mirrors to PNNL GitLab, but also supports
Incline, Decpeption and Newell.
