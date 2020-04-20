"""Hack to add per-session state to Streamlit.

Taken from https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92

Usage
-----
>>> import SessionState
>>>
>>> session_state = SessionState.get(user_name='', favorite_color='black')
>>> session_state.user_name
''
>>> session_state.user_name = 'Mary'
>>> session_state.favorite_color
'black'
Since you set user_name above, next time your script runs this will be the
result:
>>> session_state = get(user_name='', favorite_color='black')
>>> session_state.user_name
'Mary'
"""
import streamlit.ReportThread as ReportThread

from streamlit.server.Server import Server
from streamlit.ScriptRunner import RerunException
from streamlit.ScriptRequestQueue import RerunData


class SessionState(object):
    def __init__(self, **kwargs):
        """A new SessionState object.
        Parameters
        ----------
        **kwargs : any
            Default values for the session state.
        Example
        -------
        >>> session_state = SessionState(user_name='', favorite_color='black')
        >>> session_state.user_name = 'Mary'
        ''
        >>> session_state.favorite_color
        'black'
        """
        for key, val in kwargs.items():
            setattr(self, key, val)


def get(**kwargs):
    """Gets a SessionState object for the current session.
    Creates a new object if necessary.
    Parameters
    ----------
    **kwargs : any
        Default values you want to add to the session state, if we're creating a
        new one.
    Example
    -------
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    ''
    >>> session_state.user_name = 'Mary'
    >>> session_state.favorite_color
    'black'
    Since you set user_name above, next time your script runs this will be the
    result:
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    'Mary'
    """
    # Hack to get the session object from Streamlit.
    ctx = ReportThread.get_report_ctx()

    this_session = st_session()

    # Got the session object! Now let's attach some state into it.
    if not hasattr(this_session, "_custom_session_state"):
        this_session._custom_session_state = SessionState(**kwargs)

    return this_session._custom_session_state


def rerun():
    """Rerun a Streamlit app from the top!"""
    widget_states = st_session()._widget_states
    raise RerunException(RerunData(widget_states))


def st_session():
    # Hack to get the session object from Streamlit.
    ctx = ReportThread.get_report_ctx()

    this_session = None

    current_server = Server.get_current()
    if hasattr(current_server, "_session_infos"):
        # Streamlit < 0.56
        session_infos = current_server._session_infos.values()
    else:
        session_infos = current_server._session_info_by_id.values()

    for session_info in session_infos:
        s = session_info.session
        if (
            (hasattr(s, "_main_dg") and s._main_dg == ctx.main_dg)
            # Streamlit < 0.54.0
            or
            # Streamlit >= 0.54.0
            (not hasattr(s, "_main_dg") and s.enqueue == ctx.enqueue)
        ):
            this_session = s

    if this_session is None:
        raise RuntimeError(
            "Oh noes. Couldn't get your Streamlit Session object"
            "Are you doing something fancy with threads?"
        )
    return this_session


# __all__ = [get, rerun]
