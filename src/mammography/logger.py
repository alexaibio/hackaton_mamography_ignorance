import sys

from loguru import logger

from mammography.config import get_settings

FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"


def setup_logger(name: str = "mammography"):
    settings = get_settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=settings.log_level, format=FORMAT)
    logger.add(
        settings.log_dir / f"{name}.log",
        level=settings.log_level,
        format=FORMAT,
        rotation="10 MB",
        retention="7 days",
    )
    return logger
