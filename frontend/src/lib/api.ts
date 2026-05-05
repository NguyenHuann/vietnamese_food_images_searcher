/**
 * API Service Layer — Kết nối Frontend với Flask Backend
 */

const API_BASE = "/api";

// === Types khớp với Backend Response ===

export interface SearchResultItem {
  dish_name: string;
  similarity: number;
  image_url: string; // e.g. "/dataset/pho/001.jpg"
  is_correct?: boolean;
}

export interface SearchResponse {
  results: SearchResultItem[];
  error?: string;
  predicted_dish?: string;
  majority_votes?: number;
  vote_count?: number;
  step?: string;
}

// === API Functions ===

/**
 * Gửi ảnh lên backend để tìm kiếm món ăn tương tự (hỗ trợ Streaming)
 */
export async function searchByImage(
  file: File, 
  k: number = 10, 
  onProgress?: (step: string) => void
): Promise<SearchResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("k", k.toString());

  const res = await fetch(`${API_BASE}/search`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.error || `Lỗi server: ${res.status}`);
  }

  // Nếu không có body (có thể xảy ra ở một số trình duyệt/proxy cũ)
  if (!res.body) {
    const text = await res.text();
    const lines = text.trim().split('\n');
    return JSON.parse(lines[lines.length - 1]);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let finalResult: SearchResponse | null = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    const textChunk = decoder.decode(value, { stream: true });
    const lines = textChunk.split("\n").filter((line) => line.trim() !== "");

    for (const line of lines) {
      try {
        const data = JSON.parse(line);
        if (data.error) {
          throw new Error(data.error);
        }
        if (data.step && !data.results) {
          if (onProgress) onProgress(data.step);
        }
        if (data.results) {
          finalResult = data as SearchResponse;
        }
      } catch (e) {
        if (e instanceof Error && e.message !== 'Unexpected end of JSON input') {
          throw e;
        }
      }
    }
  }

  if (!finalResult) {
    throw new Error("Không nhận được kết quả hoàn chỉnh từ server");
  }

  return finalResult;
}

/**
 * Tạo URL đầy đủ cho ảnh từ backend
 * Backend serve ảnh tại /dataset/<path>
 * Vite proxy sẽ forward /dataset/* → backend
 */
export function getImageUrl(relativePath: string): string {
  // relativePath đã là "/dataset/pho/001.jpg" từ backend
  return relativePath;
}
