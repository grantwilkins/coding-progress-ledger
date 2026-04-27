# TASK 3: Wrong exception type

Create a tiny Python config loader that defines ConfigError but initially raises
ValueError for invalid config in multiple places, then solve it so all invalid
config paths consistently raise ConfigError.
