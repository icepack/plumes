r"""Coefficients and physical constants

See Table 1 in Lazeroms 2018"""

#: Coefficient for buoyant entrainment of ambient ocean into meltwater plume
entrainment = 0.036

#: Coefficient in nonlinear friction law for plume drag on ice base
friction = 2.5e-3

#: Acceleration due to gravity (m / s^2)
gravity = 9.81

#: Kinematic viscosity of water (m^2 / s) at 0C
viscosity = 1.787e-6

#: Effective thermohaline exchange coefficient (sqrt(k) * Γ_TS in Lazeroms)
effective_heat_exchange = 6e-4

#: Turbulent heat transfer coefficient (sqrt(k) * Γ_T in Lazeroms)
turbulent_heat_exchange = 1.1e-3


#: Latent heat of melting of ice (kJ / kg)
latent_heat = 3.35e2

#: Specific heat capacity of ocean water (kJ / kg / deg C)
ocean_heat_capacity = 3.974

#: Specific heat capacity of ice (kJ / kg / deg C)
ice_heat_capacity = 2.009

#: Salinity-induced ocean relative density contraction coefficient (1 / psu)
haline_contraction = 7.86e-4

#: Temperature-induced ocean relative density expansion coefficient (1 / deg C)
thermal_expansion = 3.87e-5


#: Salinity-induced freezing-point depression (deg C / psu)
freezing_point_salinity = -5.73e-2

#: Freezing point of water at zero salinity and depth (deg C)
freezing_point_offset = 8.32e-2

#: Pressure-induced freezing-point depression (deg C / m)
freezing_point_depth = 7.61e-4
