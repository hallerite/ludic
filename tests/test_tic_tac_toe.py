import pytest
from ludic_envs.envs.tic_tac_toe import TicTacToe, Action

@pytest.fixture
def game():
    return TicTacToe()

def test_reset_starts_empty(game):
    obs = game.reset(seed=42)
    flat = obs.replace('\n', '')
    assert all(c in '.XO' for c in flat)
    assert flat.count('.') in {8, 9}  # either opponent played or not
    assert game.done is False

def test_agent_side_randomization(game):
    agent_marks = set()
    for _ in range(20):
        game.reset()
        agent_marks.add(game.agent_mark)
    assert agent_marks == {'X', 'O'}

def test_illegal_move_raises(game):
    game.reset(seed=1)
    move = next(i for i, v in enumerate(game.board) if v is None)
    game._place(move, game.agent_mark)
    with pytest.raises(ValueError, match="Illegal move"):
        game._place(move, game.opponent_mark)

def test_terminal_win_detection(game):
    game.board = ['X', 'X', None,
                  'O', 'O', None,
                  None, None, None]
    game.agent_mark = 'X'
    game.opponent_mark = 'O'
    game._place(2, 'X')
    assert game._terminal_reward() == 1

def test_terminal_loss_detection(game):
    game.board = ['O', 'O', None,
                  'X', 'X', None,
                  None, None, None]
    game.agent_mark = 'X'
    game.opponent_mark = 'O'
    game._place(2, 'O')
    assert game._terminal_reward() == -1

def test_draw(game):
    game.board = ['X', 'O', 'X',
                  'X', 'O', 'O',
                  'O', 'X', None]
    game.agent_mark = 'X'
    game.opponent_mark = 'O'
    game._place(8, 'X')
    assert game._terminal_reward() == 0

def test_step_sequence_win(game):
    game.reset()
    game.agent_mark = 'X'
    game.opponent_mark = 'O'
    game.board = [None] * 9
    # force a win scenario for agent
    game.board[0] = 'X'
    game.board[1] = 'X'
    game.board[4] = 'O'
    game.board[5] = 'O'
    obs, reward, done, _ = game.step({'pos': 3}) # action index starts at 1
    assert reward == 1
    assert done is True

def test_step_sequence_draw(game):
    game.reset()
    game.agent_mark = 'X'
    game.opponent_mark = 'O'
    game.board = ['X', 'O', 'X',
                  'X', 'O', 'O',
                  'O', 'X', None]
    obs, reward, done, _ = game.step({'pos': 9}) # action index starts at 1
    assert reward == 0
    assert done is True

def test_step_after_done_raises(game):
    game.reset()
    game.board = ['X', 'X', None,
                  None, None, None,
                  None, None, None]
    game.agent_mark = 'X'
    game.opponent_mark = 'O'
    game._place(2, 'X')
    assert game._terminal_reward() == 1
    game.done = True
    with pytest.raises(RuntimeError, match="Game has ended"):
        game.step({'pos': 3})
