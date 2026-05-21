"""``python -m predictalot`` → start uvicorn serving the predictalot app."""

from .server import main

raise SystemExit(main())
