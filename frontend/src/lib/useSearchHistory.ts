import { useState, useCallback, useEffect } from 'react';
import type { HistoryItem, SearchResultItem } from '../types';
import { compressImage } from './utils';

const STORAGE_KEY = 'vn_food_search_history';
const MAX_HISTORY = 50;

function loadHistory(): HistoryItem[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveHistory(items: HistoryItem[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  } catch {
    // localStorage full — bỏ qua
  }
}

export function useSearchHistory() {
  const [history, setHistory] = useState<HistoryItem[]>(() => loadHistory());

  // Sync state → localStorage
  useEffect(() => {
    saveHistory(history);
  }, [history]);

  const addSearch = useCallback(async (
    uploadedImage: string,
    results: SearchResultItem[],
  ) => {
    if (results.length === 0) return;

    // Nén ảnh user upload thành thumbnail cực nhỏ (~10-20KB) để lưu localStorage an toàn
    let thumbnail = uploadedImage;
    try {
      thumbnail = await compressImage(uploadedImage, 300);
    } catch (e) {
      console.warn("Failed to compress thumbnail, using original (risky for localStorage)", e);
    }

    const topResult = results[0];
    const newItem: HistoryItem = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      dish_name: topResult.dish_name,
      similarity: topResult.similarity,
      image_url: topResult.image_url,
      uploadedImage: thumbnail,
      timestamp: Date.now(),
      topResults: results.slice(0, 5),
    };

    setHistory(prev => {
      const updated = [newItem, ...prev].slice(0, MAX_HISTORY);
      return updated;
    });
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  const removeItem = useCallback((id: string) => {
    setHistory(prev => prev.filter(item => item.id !== id));
  }, []);

  return { history, addSearch, clearHistory, removeItem };
}
