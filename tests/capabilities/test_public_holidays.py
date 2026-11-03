import importlib

from integrations.nager.models import PublicHoliday


lookup_module = importlib.import_module("capabilities.adapters.calendar.public_holidays_lookup")


def test_public_holidays_lookup_filters_to_region(monkeypatch) -> None:
    holidays = [
        PublicHoliday.model_validate({
            "date": "2027-02-15",
            "localName": "Family Day",
            "name": "Family Day",
            "countryCode": "CA",
            "counties": ["CA-ON", "CA-BC"],
            "types": ["Public"],
        }),
        PublicHoliday.model_validate({
            "date": "2027-11-11",
            "localName": "Remembrance Day",
            "name": "Remembrance Day",
            "countryCode": "CA",
            "counties": ["CA-AB", "CA-BC"],
            "types": ["Public"],
        }),
        PublicHoliday.model_validate({
            "date": "2027-01-01",
            "localName": "New Year's Day",
            "name": "New Year's Day",
            "countryCode": "CA",
            "counties": None,
            "types": ["Public"],
        }),
    ]
    monkeypatch.setattr(
        lookup_module._holiday_client,
        "get_public_holidays",
        lambda year, country_code: holidays,
    )

    result = lookup_module.public_holidays_lookup.invoke({
        "year": 2027,
        "country_code": "CA",
        "region_code": "CA-ON",
    })

    assert [entry.title for entry in result.evidence] == ["Family Day", "New Year's Day"]
