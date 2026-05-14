import React, { useState, useRef, useCallback } from 'react';
import { Camera, ChevronRight, Sparkles, X, UploadCloud, Loader2, Settings2, RefreshCcw, Focus } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { Link, useNavigate } from 'react-router-dom';
import Webcam from 'react-webcam';
import Cropper from 'react-easy-crop';

import { searchByImage, getImageUrl } from '../lib/api';
import { useSearchHistory } from '../lib/useSearchHistory';
import { compressImage } from '../lib/utils';
import type { SearchNavigationState } from '../types';

// HÀM TIỆN ÍCH: XỬ LÝ CẮT ẢNH BẰNG CANVAS (Gộp chung vào đây cho gọn)
const createImage = (url: string): Promise<HTMLImageElement> =>
  new Promise((resolve, reject) => {
    const image = new Image();
    image.addEventListener('load', () => resolve(image));
    image.addEventListener('error', (error) => reject(error));
    image.src = url;
  });

async function getCroppedImg(
  imageSrc: string,
  pixelCrop: { x: number; y: number; width: number; height: number },
  fileName: string = 'cropped-image.jpg'
): Promise<File | null> {
  const image = await createImage(imageSrc);
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  if (!ctx) return null;

  canvas.width = pixelCrop.width;
  canvas.height = pixelCrop.height;

  ctx.drawImage(
    image,
    pixelCrop.x,
    pixelCrop.y,
    pixelCrop.width,
    pixelCrop.height,
    0,
    0,
    pixelCrop.width,
    pixelCrop.height
  );

  return new Promise((resolve) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          resolve(null);
          return;
        }
        const file = new File([blob], fileName, { type: 'image/jpeg' });
        resolve(file);
      },
      'image/jpeg',
      0.95
    );
  });
}


export default function Home() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // --- STATE DÀNH CHO CẮT ẢNH (react-easy-crop) ---
  const [imageToCrop, setImageToCrop] = useState<string | null>(null);
  const [crop, setCrop] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [croppedAreaPixels, setCroppedAreaPixels] = useState<any>(null);
  const [isCroppingProcess, setIsCroppingProcess] = useState(false);

  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [kValue, setKValue] = useState<number>(50);
  const [loadingStep, setLoadingStep] = useState<string>('Đang khởi tạo...');

  // State quản lý Camera
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [facingMode, setFacingMode] = useState<"user" | "environment">("environment");
  const webcamRef = useRef<Webcam>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const { history, addSearch } = useSearchHistory();

  // Dọn dẹp bộ nhớ (Memory leak)
  React.useEffect(() => {
    return () => {
      if (selectedImage && selectedImage.startsWith('blob:')) {
        URL.revokeObjectURL(selectedImage);
      }
      if (imageToCrop && imageToCrop.startsWith('blob:')) {
        URL.revokeObjectURL(imageToCrop);
      }
    };
  }, [selectedImage, imageToCrop]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // Đưa ảnh vào giao diện Crop thay vì hiển thị luôn
  const processFile = (file: File) => {
    setError(null);
    const objectUrl = URL.createObjectURL(file);
    setImageToCrop(objectUrl);

    // Reset lại cấu hình crop mỗi lần mở ảnh mới
    setCrop({ x: 0, y: 0 });
    setZoom(1);
    setIsCameraOpen(false);
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const capturePhoto = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      fetch(imageSrc)
        .then(res => res.blob())
        .then(blob => {
          const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
          processFile(file);
        });
    }
  }, [webcamRef]);

  const toggleCamera = () => {
    setFacingMode(prevState => (prevState === "user" ? "environment" : "user"));
  };

  const [isDragging, setIsDragging] = React.useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      processFile(file);
    } else {
      setError('Vui lòng chỉ tải lên tệp tin hình ảnh.');
    }
  };

  // --- CÁC HÀM XỬ LÝ CỦA REACT-EASY-CROP ---
  const onCropComplete = useCallback((_croppedArea: any, croppedAreaPixels: any) => {
    setCroppedAreaPixels(croppedAreaPixels);
  }, []);

  const handleConfirmCrop = async () => {
    if (!croppedAreaPixels || !imageToCrop) return;
    setIsCroppingProcess(true);
    try {
      const croppedFile = await getCroppedImg(imageToCrop, croppedAreaPixels);
      if (croppedFile) {
        setSelectedFile(croppedFile);
        if (selectedImage && selectedImage.startsWith("blob:")) {
          URL.revokeObjectURL(selectedImage);
        }
        const objectUrl = URL.createObjectURL(croppedFile);
        setSelectedImage(objectUrl);
        setImageToCrop(null); // Đóng modal crop
      }
    } catch (e) {
      console.error("Lỗi khi cắt ảnh:", e);
      setError("Có lỗi xảy ra khi cắt ảnh.");
    } finally {
      setIsCroppingProcess(false);
    }
  };

  const handleCancelCrop = () => {
    if (imageToCrop && imageToCrop.startsWith("blob:")) {
      URL.revokeObjectURL(imageToCrop);
    }
    setImageToCrop(null);
  };


  const handleSearch = async () => {
    if (!selectedFile || !selectedImage) return;
    setIsSearching(true);
    setError(null);

    try {
      setLoadingStep('Đang gửi ảnh lên Server...');
      const data = await searchByImage(selectedFile, kValue, (step) => {
        setLoadingStep(step);
      });

      if (data.error) {
        setError(data.error);
        return;
      }

      const thumbnail = await compressImage(selectedImage, 400);
      await addSearch(thumbnail, data.results);

      const state: SearchNavigationState = {
        results: data.results,
        uploadedImage: thumbnail,
        predicted_dish: data.predicted_dish,
        majority_votes: data.majority_votes,
        vote_count: data.vote_count,
      };
      navigate('/results', { state });
    } catch (err: any) {
      setError(err.message || 'Không thể kết nối đến server.');
    } finally {
      setIsSearching(false);
    }
  };

  const recentItems = history.slice(0, 6);

  return (
    <div className="space-y-16">
      {/* MODAL CẮT ẢNH TRỰC TIẾP TẠI ĐÂY */}
      <AnimatePresence>
        {imageToCrop && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
          >
            <div className="bg-white rounded-3xl w-full max-w-lg overflow-hidden flex flex-col shadow-2xl">
              <div className="relative w-full h-[400px] bg-black">
                <Cropper
                  image={imageToCrop}
                  crop={crop}
                  zoom={zoom}
                  aspect={1} // Tỉ lệ cắt hình vuông
                  onCropChange={setCrop}
                  onZoomChange={setZoom}
                  onCropComplete={onCropComplete}
                />
              </div>

              <div className="p-6 space-y-6 bg-surface-container-lowest">
                <div className="flex items-center gap-4">
                  <span className="text-sm font-bold text-on-surface-variant">Thu phóng</span>
                  <input
                    type="range"
                    value={zoom}
                    min={1}
                    max={3}
                    step={0.1}
                    onChange={(e) => setZoom(Number(e.target.value))}
                    className="w-full h-2 bg-outline/20 rounded-lg appearance-none cursor-pointer accent-primary"
                  />
                </div>

                <div className="flex gap-4">
                  <button
                    onClick={handleCancelCrop}
                    className="flex-1 py-3 px-4 rounded-xl font-bold text-on-surface bg-surface-container-highest hover:bg-surface-container-high transition-colors"
                  >
                    Hủy
                  </button>
                  <button
                    onClick={handleConfirmCrop}
                    disabled={isCroppingProcess}
                    className="flex-1 py-3 px-4 rounded-xl font-bold text-on-primary bg-primary hover:brightness-110 transition-all disabled:opacity-50"
                  >
                    {isCroppingProcess ? "Đang xử lý..." : "Xác nhận & Tìm kiếm"}
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <section className="space-y-4 pt-12">
        <div className="flex flex-col items-center text-center space-y-6">
          <div className="space-y-1">
            <p className="micro-label">Khám phá di sản</p>
            <motion.h1
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-5xl md:text-7xl font-bold leading-[1.1] text-on-surface"
            >
              Hệ Thống Tra Cứu Món Ăn Việt Nam
            </motion.h1>
          </div>
        </div>
      </section>

      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative"
      >
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          accept="image/*"
          onChange={handleFileChange}
        />

        <div className="relative group">
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`w-full aspect-[4/5] sm:aspect-[16/9] md:aspect-[21/7] rounded-[2.5rem] overflow-hidden transition-all duration-500 relative
              ${(selectedImage || isCameraOpen) ? 'bg-black' : 'bg-surface-container-high outline-dashed outline-2 outline-outline/30 flex flex-col items-center justify-center p-6 sm:p-8 text-center'}
              ${isDragging ? 'ring-4 ring-primary ring-inset bg-primary/5' : ''}
            `}
          >
            <AnimatePresence mode="wait">
              {isCameraOpen ? (
                /* GIAO DIỆN CHỤP ẢNH TRỰC TIẾP */
                <motion.div key="camera" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="w-full h-full relative flex items-center justify-center bg-black">
                  <Webcam
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    videoConstraints={{ facingMode: facingMode }}
                    className="w-full h-full object-cover"
                  />
                  {/* Lớp phủ giao diện Camera */}
                  <div className="absolute inset-x-0 bottom-0 p-6 bg-gradient-to-t from-black/80 to-transparent flex justify-between items-center z-10">
                    <button
                      onClick={(e) => { e.stopPropagation(); setIsCameraOpen(false); }}
                      className="w-12 h-12 bg-white/20 backdrop-blur-md text-white rounded-full flex items-center justify-center hover:bg-white/40 transition"
                    >
                      <X size={20} />
                    </button>

                    <button
                      onClick={(e) => { e.stopPropagation(); capturePhoto(); }}
                      className="w-16 h-16 bg-white rounded-full border-4 border-gray-300 flex items-center justify-center hover:scale-105 active:scale-95 transition"
                    >
                      <Focus size={32} className="text-black/50" />
                    </button>

                    <button
                      onClick={(e) => { e.stopPropagation(); toggleCamera(); }}
                      className="w-12 h-12 bg-white/20 backdrop-blur-md text-white rounded-full flex items-center justify-center hover:bg-white/40 transition"
                      title="Đổi camera"
                    >
                      <RefreshCcw size={20} />
                    </button>
                  </div>
                </motion.div>
              ) : selectedImage ? (
                /* GIAO DIỆN HIỂN THỊ ẢNH ĐÃ CHỌN/ĐÃ CHỤP & CẮT XONG */
                <motion.div key="preview" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="w-full h-full relative">
                  <img src={selectedImage} alt="Preview" className="w-full h-full object-cover opacity-70" />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent" />

                  {isSearching ? (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-white space-y-6">
                      <motion.div animate={{ rotate: 360 }} transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }} className="w-20 h-20 bg-white/20 backdrop-blur-md rounded-full flex items-center justify-center border border-white/30">
                        <Loader2 size={40} />
                      </motion.div>
                      <p className="font-bold tracking-widest uppercase text-sm">{loadingStep}</p>
                    </div>
                  ) : (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-white gap-4">
                        <div className="flex gap-4">
                            <button onClick={(e) => { e.stopPropagation(); handleUploadClick(); }} className="bg-white/20 backdrop-blur-md p-4 rounded-2xl border border-white/30 hover:bg-white/40 transition-all z-10">
                                <UploadCloud size={24} />
                            </button>
                            <button onClick={(e) => { e.stopPropagation(); setIsCameraOpen(true); }} className="bg-white/20 backdrop-blur-md p-4 rounded-2xl border border-white/30 hover:bg-white/40 transition-all z-10">
                                <Camera size={24} />
                            </button>
                        </div>
                        <p className="font-bold tracking-widest uppercase text-xs z-10">Thay đổi ảnh của bạn</p>
                    </div>
                  )}

                  {!isSearching && (
                    <button
                      onClick={(e) => { e.stopPropagation(); setSelectedImage(null); setSelectedFile(null); }}
                      className="absolute top-6 right-6 w-12 h-12 bg-black/50 hover:bg-red-500 backdrop-blur-md text-white rounded-full flex items-center justify-center transition-all z-20"
                    >
                      <X size={24} />
                    </button>
                  )}
                </motion.div>
              ) : (
                /* GIAO DIỆN UPLOAD BAN ĐẦU */
                <motion.div key="upload" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-8 w-full max-w-xl mx-auto">
                  <div className="space-y-4">
                    <div className="w-20 h-20 bg-surface-container-lowest rounded-full flex items-center justify-center shadow-xl mx-auto group-hover:scale-110 transition-transform duration-500">
                        <Sparkles size={32} className="text-primary" />
                    </div>
                    <div className="space-y-2">
                        <h3 className="text-xl md:text-2xl font-bold text-on-surface">Bắt đầu khám phá ẩm thực</h3>
                        <p className="text-on-surface-variant text-sm">Tải lên từ thư viện hoặc chụp ảnh trực tiếp</p>
                    </div>
                  </div>

                  <div className="flex flex-col sm:flex-row items-center justify-center gap-4 px-6 relative z-10">
                    <button
                      onClick={(e) => { e.stopPropagation(); handleUploadClick(); }}
                      className="w-full sm:w-auto flex items-center justify-center gap-3 px-8 py-4 bg-primary text-on-primary rounded-2xl font-bold shadow-lg hover:brightness-110 active:scale-95 transition-all"
                    >
                      <UploadCloud size={20} />
                      Tải ảnh lên
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); setIsCameraOpen(true); }}
                      className="w-full sm:w-auto flex items-center justify-center gap-3 px-8 py-4 bg-surface-container-highest text-on-surface rounded-2xl font-bold border border-outline/10 shadow-md hover:bg-surface-container transition-all active:scale-95"
                    >
                      <Camera size={20} />
                      Chụp ảnh trực tiếp
                    </button>
                  </div>
                  <p className="text-[10px] uppercase tracking-widest text-on-surface-variant/50 font-bold pointer-events-none">Hoặc kéo thả ảnh vào đây</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <AnimatePresence>
            {error && (
              <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="mt-4 p-4 bg-red-50 border border-red-200 rounded-2xl text-red-700 text-center text-sm font-medium">
                ⚠️ {error}
              </motion.div>
            )}
          </AnimatePresence>

          <div className="mt-8 flex flex-col md:flex-row items-center justify-center gap-6">
            <div className={`flex items-center gap-3 bg-surface-container-high px-6 py-4 rounded-2xl transition-opacity ${isSearching ? 'opacity-50 pointer-events-none' : 'opacity-100'}`}>
              <Settings2 size={20} className="text-on-surface-variant" />
              <div className="flex flex-col">
                <label className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Số lượng kết quả</label>
                <select value={kValue} onChange={(e) => setKValue(Number(e.target.value))} className="bg-transparent text-sm font-bold text-on-surface outline-none cursor-pointer">
                  <option value={25}>25 ảnh (Nhanh)</option>
                  <option value={50}>50 ảnh (Khuyên dùng)</option>
                  <option value={75}>75 ảnh</option>
                  <option value={100}>100 ảnh</option>
                </select>
              </div>
            </div>

            <motion.div whileHover={selectedImage && !isSearching ? { scale: 1.05 } : {}} whileTap={selectedImage && !isSearching ? { scale: 0.95 } : {}} className="relative">
              <button
                onClick={handleSearch}
                disabled={!selectedImage || isSearching}
                className={`relative px-12 py-5 font-bold rounded-2xl shadow-xl transition-all duration-500 flex items-center gap-3 group
                  ${selectedImage && !isSearching ? 'bg-gradient-to-r from-primary to-primary-container text-on-primary' : 'bg-surface-container-highest text-on-surface-variant/40 grayscale'}
                `}
              >
                {isSearching ? <><Loader2 size={20} className="animate-spin" /><span>Đang xử lý...</span></> : <><Camera size={20} /><span>Bắt đầu nhận diện</span><ChevronRight size={20} className="group-hover:translate-x-2 transition-transform" /></>}
              </button>
            </motion.div>
          </div>
        </div>
      </motion.section>

      {/* Lịch sử khám phá */}
      <section className="space-y-12">
        <div className="flex flex-col items-center text-center space-y-2">
          <p className="micro-label">Thưởng thức</p>
          <h2 className="text-4xl font-bold">Khám phá gần đây</h2>
        </div>
        <div className="flex gap-8 overflow-x-auto pb-6 no-scrollbar snap-x">
          {recentItems.length > 0 ? recentItems.map((item, index) => (
            <motion.div key={item.id} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 + (index * 0.1) }} className="flex-shrink-0 w-80 snap-start group">
              <Link to="/detail" state={{ dish_name: item.dish_name, similarity: item.similarity, image_url: item.image_url, allResults: item.topResults }} className="block space-y-6 text-center">
                <div className="relative aspect-[4/5] rounded-[2rem] overflow-hidden bg-surface-container-low shadow-lg group-hover:shadow-2xl transition-all duration-500">
                  <img src={getImageUrl(item.image_url)} alt={item.dish_name} className="w-full h-full object-cover transition-transform duration-1000 group-hover:scale-110" referrerPolicy="no-referrer" />
                </div>
                <div className="space-y-1">
                  <h3 className="text-2xl font-bold text-on-surface group-hover:text-primary transition-colors">{item.dish_name}</h3>
                </div>
              </Link>
            </motion.div>
          )) : (
            <div className="w-full py-12 flex flex-col items-center justify-center text-on-surface-variant/40 border-2 border-dashed border-outline/10 rounded-[2rem]">
              <p className="font-serif italic text-lg text-center">Đang chờ di sản ẩm thực đầu tiên của bạn...</p>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}