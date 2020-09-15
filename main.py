import cv2
import numpy as np
import pandas as pd
import os
import noise_
import math
import time

if __name__ == '__main__':

    # Image Reading
    path = 'C:/Users/Juan Pablo/Im_Procesamiento'  # poner aqui la ruta de la imagen lena.png8
    image_name = 'lena.png'
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('lena_gray', image_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Noisy images generation
    lena_gauss_noisy = noise_.noise("gauss", image_gray.astype(np.float) / 255)
    lena_gauss_noisy = (255 * lena_gauss_noisy).astype(np.uint8)
    lena_sp_noisy = noise_.noise("s&p", image_gray.astype(np.float) / 255)
    lena_sp_noisy = (255 * lena_gauss_noisy).astype(np.uint8)
    cv2.imshow('lena_gauss_noisy | lena_s&p_noisy', cv2.hconcat([lena_gauss_noisy, lena_sp_noisy]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Gauss Filtering
    N = 7
    start_time = time.time()
    #  for lena_gauss_noisy
    lena_gauss_noisy_lp = cv2.GaussianBlur(lena_gauss_noisy, (N, N), 1.5, 1.5)
    gauss_lp_time = time.time() - start_time
    print('gauss_lp_time:', gauss_lp_time)
    lena_gauss_noise_lp = abs(lena_gauss_noisy - lena_gauss_noisy_lp)
    cv2.imshow('lena_gauss_noisy_lp | lena_gauss_noise_lp', cv2.hconcat([lena_gauss_noisy_lp, lena_gauss_noise_lp]))
    ECM_gauss_lp = math.sqrt(np.square(np.subtract(image_gray, lena_gauss_noise_lp)).mean())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    start_time = time.time()
    #  for lena_sp_noisy
    lena_sp_noisy_lp = cv2.GaussianBlur(lena_sp_noisy, (N, N), 1.5, 1.5)
    sp_lp_time = time.time() - start_time
    print('sp_lp_time:', sp_lp_time)
    lena_sp_noise_lp = abs(lena_sp_noisy - lena_sp_noisy_lp)
    cv2.imshow('lena_sp_noisy_lp | lena_sp_noise_lp', cv2.hconcat([lena_sp_noisy_lp, lena_sp_noise_lp]))
    ECM_sp_lp = math.sqrt(np.square(np.subtract(image_gray, lena_sp_noise_lp)).mean())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Median Filtering
    start_time = time.time()
    #  for lena_gauss_noisy
    lena_gauss_noisy_median = cv2.medianBlur(lena_gauss_noisy, 7)
    gauss_median_time = time.time() - start_time
    print('gauss_median_time:', gauss_median_time)
    lena_gauss_noise_median = abs(lena_gauss_noisy - lena_gauss_noisy_median)
    cv2.imshow('lena_gauss_noisy_median | lena_gauss_noise_median', cv2.hconcat([lena_gauss_noisy_median, lena_gauss_noise_median]))
    ECM_gauss_median = math.sqrt(np.square(np.subtract(image_gray, lena_gauss_noise_median)).mean())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    start_time = time.time()
    #  for lena_sp_noisy
    lena_sp_noisy_median = cv2.medianBlur(lena_sp_noisy, 7)
    sp_median_time = time.time() - start_time
    print('sp_median_time:', sp_median_time)
    lena_sp_noise_median = abs(lena_sp_noisy - lena_sp_noisy_median)
    cv2.imshow('lena_sp_noisy_median | lena_sp_noise_median', cv2.hconcat([lena_sp_noisy_median, lena_sp_noise_median]))
    ECM_sp_median = math.sqrt(np.square(np.subtract(image_gray, lena_sp_noise_median)).mean())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Bilateral Filtering
    start_time = time.time()
    #  for lena_gauss_noisy
    lena_gauss_noisy_bilateral = cv2.bilateralFilter(lena_gauss_noisy, 15, 25, 25)
    gauss_bilateral_time = time.time() - start_time
    print('gauss_bilateral_time:', gauss_bilateral_time)
    lena_gauss_noise_bilateral = abs(lena_gauss_noisy - lena_gauss_noisy_bilateral)
    cv2.imshow('lena_gauss_noisy_bilateral | lena_gauss_noise_bilateral', cv2.hconcat([lena_gauss_noisy_bilateral, lena_gauss_noise_bilateral]))
    ECM_gauss_bilateral = math.sqrt(np.square(np.subtract(image_gray, lena_gauss_noise_bilateral)).mean())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    start_time = time.time()
    #  for lena_sp_noisy
    lena_sp_noisy_bilateral = cv2.bilateralFilter(lena_sp_noisy, 15, 25, 25)
    sp_bilateral_time = time.time() - start_time
    print('sp_bilateral_time:', sp_bilateral_time)
    lena_sp_noise_bilateral = abs(lena_sp_noisy - lena_sp_noisy_bilateral)
    cv2.imshow('lena_sp_noisy_bilateral | lena_sp_noise_bilateral', cv2.hconcat([lena_sp_noisy_bilateral, lena_sp_noise_bilateral]))
    ECM_sp_bilateral = math.sqrt(np.square(np.subtract(image_gray, lena_sp_noise_bilateral)).mean())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # nlm Filtering
    start_time = time.time()
    #  for lena_gauss_noisy
    lena_gauss_noisy_nlm = cv2.fastNlMeansDenoising(lena_gauss_noisy, 5, 15, 25)
    gauss_nlm_time = time.time() - start_time
    print('gauss_nlm_time:', gauss_nlm_time)
    lena_gauss_noise_nlm = abs(lena_gauss_noisy - lena_gauss_noisy_nlm)
    cv2.imshow('lena_gauss_noisy_nlm | lena_gauss_noise_nlm', cv2.hconcat([lena_gauss_noisy_nlm, lena_gauss_noise_nlm]))
    ECM_gauss_nlm = math.sqrt(np.square(np.subtract(image_gray, lena_gauss_noise_nlm)).mean())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    start_time = time.time()
    #  for lena_sp_noisy
    lena_sp_noisy_nlm = cv2.fastNlMeansDenoising(lena_sp_noisy, 5, 15, 25)
    sp_nlm_time = time.time() - start_time
    print('sp_nlm_time:', sp_nlm_time)
    lena_sp_noise_nlm = abs(lena_sp_noisy - lena_sp_noisy_nlm)
    cv2.imshow('lena_sp_noisy_nlm | lena_sp_noise_nlm', cv2.hconcat([lena_sp_noisy_nlm, lena_sp_noise_nlm]))
    ECM_sp_nlm = math.sqrt(np.square(np.subtract(image_gray, lena_sp_noise_nlm)).mean())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Data Analysis

    # Filtering times table
    mintime1 = min(np.array([gauss_lp_time, gauss_median_time, gauss_bilateral_time, gauss_nlm_time]))
    mintime2 = min(np.array([sp_lp_time, sp_median_time, sp_bilateral_time, sp_nlm_time]))
    Data_time = {'Filter Type': ['Gaussian LP', 'Median', 'Bilateral', 'NLM'],
                 'Times Lena Gauss Noise': [gauss_lp_time, gauss_median_time, gauss_bilateral_time, gauss_nlm_time]/mintime1,
                 'Times Lena S&P Noise': [sp_lp_time, sp_median_time, sp_bilateral_time, sp_nlm_time]/mintime2}
    df = pd.DataFrame(Data_time)
    print('Filtering times table\n')
    print(pd.DataFrame(Data_time), '\n')

    # ECM of filtering table
    Data_ECM = {'Filter Type': ['Gaussian LP', 'Median', 'Bilateral', 'NLM'],
                 'ECM Lena Gauss Noise': [ECM_gauss_lp, ECM_gauss_median, ECM_gauss_bilateral, ECM_gauss_nlm],
                 'ECM Lena S&P Noise': [ECM_sp_lp, ECM_sp_median, ECM_sp_bilateral, ECM_sp_nlm]}
    print('ECM of filtering table\n')
    print(pd.DataFrame(Data_ECM), '\n')