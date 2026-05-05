import { useState, useCallback, useEffect } from 'react';

const STORAGE_KEY = 'vn_food_favorites';

export interface FavoriteItem {
  dish_name: string;
  image_url: string;
  timestamp: number;
}

function loadFavorites(): FavoriteItem[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveFavorites(items: FavoriteItem[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  } catch {
    // localStorage full
  }
}

export function useFavorites() {
  const [favorites, setFavorites] = useState<FavoriteItem[]>(() => loadFavorites());

  useEffect(() => {
    saveFavorites(favorites);
  }, [favorites]);

  const toggleFavorite = useCallback((item: FavoriteItem) => {
    setFavorites(prev => {
      const exists = prev.find(f => f.dish_name === item.dish_name);
      if (exists) {
        return prev.filter(f => f.dish_name !== item.dish_name);
      }
      return [item, ...prev];
    });
  }, []);

  const isFavorite = useCallback((dish_name: string) => {
    return favorites.some(f => f.dish_name === dish_name);
  }, [favorites]);

  return { favorites, toggleFavorite, isFavorite };
}
