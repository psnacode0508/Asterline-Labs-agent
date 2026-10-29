"""SQLite-backed storage for user-specific preferences and configurations."""

import sqlite3
from typing import Dict


class UserMemoryStore:
    """Persist personal user context without storing company documentation."""

    def __init__(self, db_path: str = "asterline_memory.db"):
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        """Create the user-memory table if it has not already been created."""
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memory (
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, key)
            )
            """
        )
        self.connection.commit()

    def get_user_memory(self, user_id: str) -> Dict[str, str]:
        """Return all personal memory entries belonging to ``user_id`` only."""
        cursor = self.connection.execute(
            "SELECT key, value FROM user_memory WHERE user_id = ? ORDER BY key",
            (user_id,),
        )
        return {row["key"]: row["value"] for row in cursor.fetchall()}

    def set_memory(self, user_id: str, key: str, value: str) -> None:
        """Create or update one personal memory entry for the given user."""
        self.connection.execute(
            """
            INSERT INTO user_memory (user_id, key, value, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, key, value),
        )
        self.connection.commit()

    def forget_memory(self, user_id: str, key: str) -> bool:
        """Delete one personal entry for the user and report whether it existed."""
        cursor = self.connection.execute(
            "DELETE FROM user_memory WHERE user_id = ? AND key = ?",
            (user_id, key),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def clear_user_memory(self, user_id: str) -> int:
        """Delete all personal entries for the given user and return their count."""
        cursor = self.connection.execute(
            "DELETE FROM user_memory WHERE user_id = ?",
            (user_id,),
        )
        self.connection.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close the SQLite connection when the store is no longer needed."""
        self.connection.close()


if __name__ == "__main__":
    store = UserMemoryStore()
    alice_id = "user_alice"
    bob_id = "user_bob"
    memory_key = "workspace"

    store.clear_user_memory(alice_id)
    store.clear_user_memory(bob_id)
    store.set_memory(alice_id, memory_key, "Asterline production workspace")

    assert store.get_user_memory(alice_id)[memory_key] == "Asterline production workspace"
    assert memory_key not in store.get_user_memory(bob_id)
    assert store.forget_memory(alice_id, memory_key) is True
    assert memory_key not in store.get_user_memory(alice_id)

    store.close()
    print("UserMemoryStore self-test passed.")
