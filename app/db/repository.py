
"""
Report Repository

Responsibilities:
- Read-only database access
- Database schema inspection
- Parameterized SELECT queries
- Report lookup contracts
"""

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.db.connection import readonly_transaction


logger = logging.getLogger(__name__)


class ReportRepository:
    """
    Repository for report database access.

    All database operations must use read-only transactions.
    """

    def __init__(
        self,
        engine: Engine,
    ):
        self.engine = engine

    # ========================================================
    # 1. Database Connectivity
    # ========================================================

    def health_check(self) -> dict:

        with readonly_transaction(
            self.engine
        ) as connection:

            result = connection.execute(
                text("SELECT 1")
            )

            value = result.scalar_one()

        return {
            "status": "healthy",
            "result": value,
        }

    # ========================================================
    # 2. Database Schema Inspection
    # ========================================================

    def list_tables(
        self,
        schema_name: str = "public",
    ) -> list[dict]:
        """
        List tables and views in a specified schema.
        """

        query = text(
            """
            SELECT
                table_schema,
                table_name,
                table_type
            FROM information_schema.tables
            WHERE table_schema = :schema_name
            ORDER BY table_name
            """
        )

        with readonly_transaction(
            self.engine
        ) as connection:

            result = connection.execute(
                query,
                {
                    "schema_name": schema_name,
                },
            )

            return [
                dict(row)
                for row in result.mappings().all()
            ]

    # ========================================================
    # 3. Table Column Inspection
    # ========================================================

    def get_table_columns(
        self,
        table_name: str,
        schema_name: str = "public",
    ) -> list[dict]:
        """
        Retrieve column definitions from information_schema.

        Table and schema names are bound as query parameters.
        """

        query = text(
            """
            SELECT
                column_name,
                data_type,
                udt_name,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_schema = :schema_name
              AND table_name = :table_name
            ORDER BY ordinal_position
            """
        )

        with readonly_transaction(
            self.engine
        ) as connection:

            result = connection.execute(
                query,
                {
                    "schema_name": schema_name,
                    "table_name": table_name,
                },
            )

            return [
                dict(row)
                for row in result.mappings().all()
            ]

    # ========================================================
    # 4. Primary Key Inspection
    # ========================================================

    def get_primary_keys(
        self,
        table_name: str,
        schema_name: str = "public",
    ) -> list[str]:

        query = text(
            """
            SELECT
                kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema = :schema_name
              AND tc.table_name = :table_name
            ORDER BY kcu.ordinal_position
            """
        )

        with readonly_transaction(
            self.engine
        ) as connection:

            result = connection.execute(
                query,
                {
                    "schema_name": schema_name,
                    "table_name": table_name,
                },
            )

            return [
                row[0]
                for row in result.fetchall()
            ]

    # ========================================================
    # 5. Database Metadata Summary
    # ========================================================

    def get_schema_summary(
        self,
        schema_name: str = "public",
    ) -> list[dict]:
        """
        Return tables with their columns and primary keys.

        Useful during initial project development.
        """

        tables = self.list_tables(schema_name)

        summary = []

        for table in tables:

            table_name = table["table_name"]

            columns = self.get_table_columns(
                table_name,
                schema_name,
            )

            primary_keys = self.get_primary_keys(
                table_name,
                schema_name,
            )

            summary.append(
                {
                    "schema": schema_name,
                    "table": table_name,
                    "table_type": table["table_type"],
                    "primary_keys": primary_keys,
                    "columns": columns,
                }
            )

        return summary
