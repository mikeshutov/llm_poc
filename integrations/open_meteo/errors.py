class WeatherClientError(RuntimeError):
    pass


class WeatherGeocodingError(WeatherClientError):
    pass


class WeatherArchiveError(WeatherClientError):
    pass


class WeatherNotFoundError(WeatherClientError):
    pass
