"""Program library: ordinary importable modules copied
into workspaces that want them. No module reads a config or env var to
decide its behavior; none is mandatory — a program may use all, some, or
none of them. Orchestration is control flow in the program's own main.py,
never in here."""
