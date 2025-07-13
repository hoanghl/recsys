import psycopg as pg
from loguru import logger
from psycopg.rows import dict_row

from src import config


def db_connect(func):
    def func_inner(*args, **kwargs):
        try:
            with pg.connect(
                user=config.USER,
                password=config.PWD,
                host=config.HOST,
                port=config.PORT,
                dbname=config.DB_NAME,
                row_factory=dict_row,
            ) as conn:
                with conn.cursor() as cur:
                    return func(cur=cur, conn=conn, *args, **kwargs)

        except pg.OperationalError as e:
            logger.error(f"Connect to DB got error: {e}")

            return None

    return func_inner


@db_connect
def fetch_similar_items(**kwargs) -> list[dict]:
    query = """
        select
            id
            ,item_type
            ,location
            ,dt_added
            ,embedding <=> {value} as dist_cosine
        from
            {table}
        order by embedding <=> {value}
        limit {topk}
        ;
    """

    # Trigger query
    cur = kwargs["cur"]

    stmt = pg.sql.SQL(query).format(
        table=pg.sql.Identifier(config.TABLE_ITEMS),
        value=pg.sql.Literal(kwargs.get("text_embd")),
        topk=pg.sql.Literal(kwargs.get("topk")),
    )
    # logger.debug(stmt.as_string())

    cur.execute(stmt)

    ret = cur.fetchall()

    # logger.debug(ret)

    return ret


@db_connect
def fetch_item(**kwargs) -> dict:
    query = """
        select
            id
            ,item_type
            ,location
        from
            {table}
        where 1=1
            and id = {item_id}
        ;
    """

    # Trigger query
    cur = kwargs["cur"]

    stmt = pg.sql.SQL(query).format(
        table=pg.sql.Identifier(config.TABLE_ITEMS),
        item_id=pg.sql.Literal(kwargs.get("item_id")),
    )
    # logger.debug(stmt.as_string())

    cur.execute(stmt)

    ret = cur.fetchone()

    return ret


@db_connect
def insert_items(**kwargs):
    cols = ["item_type", "location", "embedding"]

    query = """
        insert into {table}
            ({cols})
        values ({values})
        ;
    """

    # Trigger query
    cur = kwargs["cur"]
    values = kwargs.get("values")

    stmt = pg.sql.SQL(query).format(
        table=pg.sql.Identifier(config.TABLE_ITEMS),
        cols=pg.sql.SQL(", ").join(map(pg.sql.Identifier, cols)),
        values=pg.sql.SQL(", ").join(pg.sql.Placeholder() * len(values[0])),
    )
    # logger.debug(stmt.as_string())

    cur.execute(stmt)

    ret = cur.fetchall()

    return ret
