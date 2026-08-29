from rendering.cards import _ingredient_lines


def test_ingredient_lines_include_measures_when_available() -> None:
    assert _ingredient_lines(
        [
            {"name": "Flour", "measure": "2 cups"},
            {"name": "Eggs", "measure": None},
            {"name": "", "measure": "1 tsp"},
        ]
    ) == ["2 cups Flour", "Eggs"]
