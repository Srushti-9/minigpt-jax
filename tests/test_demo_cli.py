import sys
from unittest.mock import MagicMock, patch


def test_demo_launch_keyboard_interrupt_does_not_propagate():
    mock_demo = MagicMock()
    mock_demo.launch.side_effect = KeyboardInterrupt

    mock_interface = MagicMock(return_value=mock_demo)

    checkpoint_path = "dummy_checkpoint"

    with patch("sys.argv", ["minigpt demo", "--checkpoint", checkpoint_path]):
        with patch("minigpt.cli.demo_cli.gr.Interface", mock_interface):
            with patch("minigpt.cli.demo_cli.get_tokenizer") as mock_tok:
                with patch("minigpt.cli.demo_cli.MiniGPT"):
                    with patch("minigpt.cli.demo_cli.load_checkpoint"):
                        mock_tok.return_value = MagicMock(n_vocab=512)
                        from minigpt.cli.demo_cli import main
                        main()

    mock_demo.launch.assert_called_once()
