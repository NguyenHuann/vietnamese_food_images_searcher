import { ChevronRight, Trash2, Calendar } from 'lucide-react';
import { motion } from 'motion/react';
import { Link } from 'react-router-dom';
import { useSearchHistory } from '../lib/useSearchHistory';
import { getImageUrl } from '../lib/api';
import type { HistoryItem } from '../types';

export default function History() {
  const { history, clearHistory, removeItem } = useSearchHistory();

  // Nhóm theo ngày
  const grouped = history.reduce<Record<string, HistoryItem[]>>((acc, item) => {
    const date = new Date(item.timestamp).toLocaleDateString('vi-VN', {
      weekday: 'long',
      day: 'numeric',
      month: 'long',
      year: 'numeric',
    });
    if (!acc[date]) acc[date] = [];
    acc[date].push(item);
    return acc;
  }, {});

  const groupEntries: [string, HistoryItem[]][] = Object.entries(grouped);

  return (
    <div className="space-y-12">
      <section className="space-y-4">
        <p className="micro-label">Kho lưu trữ</p>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight">Lịch sử khám phá</h1>
        <p className="text-on-surface-variant max-w-xl text-lg font-serif italic">
          Nơi lưu giữ những hành trình giải mã ẩm thực và các công thức di sản mà bạn đã tìm thấy.
        </p>
      </section>

      <div className="space-y-12">
        {groupEntries.length > 0 ? groupEntries.map(([date, items], groupIndex) => (
          <section key={date} className="space-y-8">
            <div className="flex items-center gap-6">
              <span className="text-sm font-bold uppercase tracking-[0.2em] text-primary">
                {date}
              </span>
              <div className="h-px flex-grow bg-outline/10" />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {items.map((item, itemIndex) => (
                <motion.div
                  key={item.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: (groupIndex * 0.2) + (itemIndex * 0.1) }}
                >
                  <Link 
                    to="/detail"
                    state={{
                      dish_name: item.dish_name,
                      similarity: item.similarity,
                      image_url: item.image_url,
                      allResults: item.topResults,
                    }}
                    className="group flex items-center gap-8 p-6 rounded-[2rem] bg-surface-container-low hover:bg-white hover:shadow-2xl transition-all duration-500 border border-outline/5 active:scale-[0.98]"
                  >
                    <div className="relative w-32 h-32 flex-shrink-0 rounded-[1.5rem] overflow-hidden shadow-lg">
                      <img 
                        src={getImageUrl(item.image_url)} 
                        alt={item.dish_name} 
                        className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
                        referrerPolicy="no-referrer"
                        loading="lazy"
                      />
                    </div>
                    
                    <div className="flex-grow space-y-3 min-w-0">
                      <h3 className="text-2xl font-bold text-on-surface group-hover:text-primary transition-colors truncate">
                        {item.dish_name}
                      </h3>
                      <p className="text-sm text-on-surface-variant font-serif italic">
                        {new Date(item.timestamp).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })}
                      </p>
                      <div className="flex gap-2">
                        <span className="px-3 py-1 border border-outline/10 rounded-full text-[9px] font-bold uppercase tracking-widest text-on-surface/40">
                          {(item.similarity * 100).toFixed(0)}% khớp
                        </span>
                      </div>
                    </div>

                    <div className="flex flex-col items-center gap-4 flex-shrink-0">
                      <button
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          if (confirm('Bạn có chắc muốn xóa mục này khỏi lịch sử?')) {
                            removeItem(item.id);
                          }
                        }}
                        className="p-2 text-on-surface-variant/20 hover:text-red-500 hover:bg-red-50 rounded-full transition-all"
                        title="Xóa mục này"
                      >
                        <Trash2 size={18} />
                      </button>
                      <div className="text-primary/20 group-hover:text-primary transition-colors">
                        <ChevronRight size={24} />
                      </div>
                    </div>
                  </Link>
                </motion.div>
              ))}
            </div>
          </section>
        )) : (
          <div className="py-24 flex flex-col items-center justify-center space-y-6 border-2 border-dashed border-outline/10 rounded-[3rem] bg-surface-container-low/30">
            <div className="w-24 h-24 bg-surface-container-high rounded-full flex items-center justify-center text-primary/10">
              <Calendar size={48} />
            </div>
            <div className="text-center space-y-2">
              <h3 className="text-2xl font-bold">Chưa có lịch sử</h3>
              <p className="text-on-surface-variant font-serif italic">Hãy bắt đầu hành trình khám phá ẩm thực đầu tiên của bạn.</p>
            </div>
            <Link to="/" className="px-10 py-4 bg-on-surface text-white rounded-full font-bold uppercase text-[10px] tracking-[0.2em] hover:bg-primary transition-all">
              Bắt đầu ngay
            </Link>
          </div>
        )}
      </div>

      {history.length > 0 && (
        <div className="pt-12 flex justify-center">
          <button 
            onClick={clearHistory}
            className="flex items-center gap-3 text-on-surface-variant hover:text-red-500 transition-colors px-12 py-4 rounded-full border border-outline/20 font-bold text-[10px] uppercase tracking-[0.2em] shadow-sm hover:shadow-xl group hover:border-red-200"
          >
            <Trash2 size={16} className="group-hover:rotate-12 transition-transform" />
            Xóa toàn bộ lịch sử
          </button>
        </div>
      )}
    </div>
  );
}
