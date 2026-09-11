"""The keypress reducer: pure, so it can be exercised a keystroke at a time."""

from burn.analysis import tabulate
from burn.keys import apply
from burn.model import Usage
from burn.state import FILTER, HELP, MAX_WINDOW, MIN_WINDOW, TABLE, TOOLS, TURNS, View, cursor


def table(snapshot, call, sessions=("aaa", "bbb", "ccc")):
    calls = [call(session=s, usage=Usage(input=n)) for n, s in enumerate(sessions, 1)]
    return sorted(tabulate(snapshot(calls=calls)), key=lambda r: -r.weight)


def press(view, table, *keys):
    for key in keys:
        view = apply(key, view, table)
        if view is None:
            return None
    return view


def test_q_and_ctrl_c_quit(snapshot, call):
    rows = table(snapshot, call)
    assert apply("q", View(), rows) is None
    assert apply("\x03", View(), rows) is None


def test_an_unbound_key_changes_nothing(snapshot, call):
    view = View()
    assert apply("z", view, table(snapshot, call)) == view


def test_moving_down_selects_the_next_row(snapshot, call):
    rows = table(snapshot, call)
    moved = press(View(), rows, "j")
    assert moved.selected == rows[1].session
    assert cursor(moved, rows) == 1


def test_the_cursor_stops_at_both_ends(snapshot, call):
    rows = table(snapshot, call)
    assert press(View(), rows, "k", "k").selected == rows[0].session
    assert press(View(), rows, *"jjjjj").selected == rows[-1].session


def test_arrow_keys_move_like_j_and_k(snapshot, call):
    rows = table(snapshot, call)
    assert press(View(), rows, "\x1b[B").selected == rows[1].session
    assert press(View(), rows, "\x1b[B", "\x1b[A").selected == rows[0].session


def test_the_selection_follows_its_session_when_the_sort_flips(snapshot, call):
    # This is why the selection is a session id and not a row index: reversing
    # the sort must not slide the cursor onto a different session.
    rows = table(snapshot, call)
    view = press(View(), rows, "j")
    chosen = view.selected
    flipped = sorted(rows, key=lambda r: r.weight)
    assert press(view, rows, "r").selected == chosen
    assert cursor(view, flipped) == len(rows) - 2


def test_moving_in_an_empty_table_selects_nothing():
    assert press(View(), [], "j", "k", "\r").selected is None


def test_enter_toggles_the_zoom_pane(snapshot, call):
    rows = table(snapshot, call)
    zoomed = press(View(), rows, "\r")
    assert zoomed.zoomed and zoomed.selected == rows[0].session
    assert not press(zoomed, rows, "\r").zoomed


def test_t_swaps_the_zoom_pane(snapshot, call):
    rows = table(snapshot, call)
    assert press(View(), rows, "t").panel == TURNS
    assert press(View(), rows, "t", "t").panel == TOOLS


def test_s_cycles_the_sort_column_both_ways(snapshot, call):
    rows = table(snapshot, call)
    assert press(View(), rows, "s").sort == "rate"
    assert press(View(), rows, "s", "S").sort == "weight"
    assert press(View(), rows, "S").sort == "session"  # wraps off the front


def test_a_cycles_both_agents_then_back(snapshot, call):
    rows = table(snapshot, call)
    assert [press(View(), rows, *"a" * n).source for n in (1, 2, 3)] == ["cc", "cx", None]


def test_the_window_doubles_and_halves_within_bounds(snapshot, call):
    rows = table(snapshot, call)
    assert press(View(window=300), rows, "+").window == 600
    assert press(View(window=300), rows, "-").window == 150
    assert press(View(window=4), rows, "-").window == MIN_WINDOW
    assert press(View(window=MAX_WINDOW), rows, "+").window == MAX_WINDOW


def test_filter_mode_takes_text_and_enter_accepts_it(snapshot, call):
    rows = table(snapshot, call)
    typed = press(View(), rows, "/", "b", "u", "r", "n")
    assert typed.mode == FILTER and typed.draft == "burn"
    accepted = press(typed, rows, "\r")
    assert accepted.needle == "burn" and accepted.mode == TABLE


def test_escape_abandons_the_draft_and_keeps_the_old_filter(snapshot, call):
    rows = table(snapshot, call)
    typed = press(View(needle="old"), rows, "/", "n", "e", "w")
    assert press(typed, rows, "\x1b").needle == "old"


def test_backspace_deletes_one_character(snapshot, call):
    rows = table(snapshot, call)
    assert press(View(), rows, "/", "a", "b", "\x7f").draft == "a"


def test_q_is_a_letter_while_filtering_not_a_quit(snapshot, call):
    rows = table(snapshot, call)
    typing = press(View(), rows, "/", "q")
    assert typing is not None and typing.draft == "q"


def test_any_key_dismisses_help(snapshot, call):
    rows = table(snapshot, call)
    assert press(View(), rows, "h").mode == HELP
    assert press(View(), rows, "h", "x").mode == TABLE


def test_c_clears_the_filter_zoom_and_agent(snapshot, call):
    rows = table(snapshot, call)
    cluttered = View(needle="x", zoomed=True, source="cc")
    cleared = press(cluttered, rows, "c")
    assert (cleared.needle, cleared.zoomed, cleared.source) == ("", False, None)


def test_space_pauses_and_resumes(snapshot, call):
    rows = table(snapshot, call)
    assert press(View(), rows, " ").paused
    assert not press(View(), rows, " ", " ").paused


def test_a_view_is_never_mutated_in_place(snapshot, call):
    rows = table(snapshot, call)
    view = View()
    press(view, rows, "j", "s", "r", "+", "a")
    assert view == View(started=view.started)
