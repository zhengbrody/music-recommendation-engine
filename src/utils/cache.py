"""
Caching utilities using Redis.
"""

import json
import redis
from typing import Any, Optional
from config.config import Config


class CacheManager:
    """Manage caching operations with Redis."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize cache manager.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()

        try:
            self.client = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                decode_responses=True
            )
            # Test connection
            self.client.ping()
            self.enabled = True
            print("Redis cache connected successfully")
        except (redis.ConnectionError, redis.RedisError) as e:
            print(f"Redis connection failed: {e}. Caching disabled.")
            self.enabled = False
            self.client = None

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.enabled:
            return None

        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            print(f"Cache get error: {e}")

        return None

    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            expire: Expiration time in seconds (uses config default if None)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            expire = expire or self.config.CACHE_EXPIRE
            serialized = json.dumps(value)
            self.client.setex(key, expire, serialized)
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            self.client.delete(key)
            return True
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False

    def clear_all(self) -> bool:
        """
        Clear all cache entries.

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            self.client.flushdb()
            return True
        except Exception as e:
            print(f"Cache clear error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        if not self.enabled:
            return False

        try:
            return self.client.exists(key) > 0
        except Exception as e:
            print(f"Cache exists error: {e}")
            return False
