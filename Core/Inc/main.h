/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "datatypes.h"
#include "utils.h"
#include "xnor_base.h"
#include "xnor_fc.h"
#include "3pxnet_fc.h"
#include "3pxnet_cn.h"
#include "xnor_fc.h"
#include "batch.h"
#include "bwn_dense_cn.h"
#include "conv1_weight.h" 
#include "conv1_bias.h"
#include "conv2_weight.h" 
#include "conv3_weight.h" 
#include "conv4_weight.h" 
#include "bn4_running_mean.h" 
#include "bn4_running_var.h" 
#include "bn4_bias.h" 
#include "bn4_weight.h" 
#include "bn1.h" 
#include "bn2.h" 
#include "bn3.h" 
#include "image.h"
/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */
static int8_t l1_act[] = IMAGES ; 
static uint8_t   labels[] = LABELS; 
#define C1KXY 5
#define C1XY   32
#define C1Z   3
#define C1KZ 32
#define C1PD 2
#define C1PL 2
//static int8_t l1wght[] = _conv1_weight ;
static q15_t l1wght[] = _conv1_weight ;
static q15_t l1bias[] = _conv1_bias ;
#define C2KXY 5
#define C2XY ((2*C1PD+C1XY-C1KXY+1)/C1PL) 
#define C2Z 32
#define C2KZ 32
static pckDtype l2act_bin[C2XY*C2XY*C2Z/pckWdt*2]; 
#define C2PD 2
#define C2PL 2
static pckDtype l2wght[] = _conv2_weight ;
#define C3KXY 5
#define C3XY ((2*C2PD+C2XY-C2KXY+1)/C2PL) 
#define C3Z 32
#define C3KZ 32
static pckDtype l3act_bin[C3XY*C3XY*C3Z/pckWdt*2]; 
#define C3PD 2
#define C3PL 2
static pckDtype l3wght[] = _conv3_weight ;
#define C4KXY 4
#define C4XY ((2*C3PD+C3XY-C3KXY+1)/C3PL) 
#define C4Z 32
#define C4KZ 10
static pckDtype l4act_bin[C4XY*C4XY*C4Z/pckWdt*2]; 
#define C4PD 0
#define C4PL 1 
static pckDtype l4wght[] = _conv4_weight ;
static float output[10]; 
static pckDtype bn1thr[] = bn1_thresh ; 
static pckDtype bn1sign[] = bn1_sign ; 
static pckDtype bn1offset[] = bn1_offset ; 
static pckDtype bn2thr[] = bn2_thresh ; 
static pckDtype bn2sign[] = bn2_sign ; 
static pckDtype bn2offset[] = bn2_offset ; 
static pckDtype bn3thr[] = bn3_thresh ; 
static pckDtype bn3sign[] = bn3_sign ; 
static pckDtype bn3offset[] = bn3_offset ; 
static bnDtype bn4mean[] = _bn4_running_mean ; 
static bnDtype bn4var[] = _bn4_running_var ; 
static bnDtype bn4gamma[] = _bn4_weight ; 
static bnDtype bn4beta[] = _bn4_bias ; 
/*static int8_t l1_act[] = IMAGES ; 
static uint8_t   labels[] = LABELS; 
#define C1KXY 5
#define C1XY   32
#define C1Z   3
#define C1KZ 32
#define C1PD 2
#define C1PL 2
static int8_t l1wght[] = _conv1_weight ;
#define C2KXY 5
#define C2XY ((2*C1PD+C1XY-C1KXY+1)/C1PL) 
#define C2Z 32
#define C2KZ 32
static pckDtype l2act_bin[C2XY*C2XY*C2Z/pckWdt*2]; 
#define C2PD 2
#define C2PL 2
#define C2NPI 96
static uint8_t l2ind[] = _conv2_weight_indices ;
static pckDtype l2wght[] = _conv2_weight ;
#define C3KXY 5
#define C3XY ((2*C2PD+C2XY-C2KXY+1)/C2PL) 
#define C3Z 32
#define C3KZ 32
static pckDtype l3act_bin[C3XY*C3XY*C3Z/pckWdt*2]; 
#define C3PD 2
#define C3PL 2
#define C3NPI 96
static uint8_t l3ind[] = _conv3_weight_indices ;
static pckDtype l3wght[] = _conv3_weight ;
#define C4KXY 4
#define C4XY ((2*C3PD+C3XY-C3KXY+1)/C3PL) 
#define C4Z 32
#define C4KZ 10
static pckDtype l4act_bin[C4XY*C4XY*C4Z/pckWdt*2]; 
#define C4PD 0
#define C4PL 1 
#define C4NPI 64
static uint8_t l4ind[] = _fc1_weight_indices ;
static pckDtype l4wght[] = _fc1_weight ;
static float output[10]; 
static pckDtype bn1thr[] = bn1_thresh ; 
static pckDtype bn1sign[] = bn1_sign ; 
static pckDtype bn1offset[] = bn1_offset ; 
static pckDtype bn2thr[] = bn2_thresh ; 
static pckDtype bn2sign[] = bn2_sign ; 
static pckDtype bn2offset[] = bn2_offset ; 
static pckDtype bn3thr[] = bn3_thresh ; 
static pckDtype bn3sign[] = bn3_sign ; 
static pckDtype bn3offset[] = bn3_offset ; 
static bnDtype bn4mean[] = _bnfc1_running_mean ; 
static bnDtype bn4var[] = _bnfc1_running_var ; 
static bnDtype bn4gamma[] = _bnfc1_weight ; 
static bnDtype bn4beta[] = _bnfc1_bias ; */
/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define B1_Pin GPIO_PIN_13
#define B1_GPIO_Port GPIOC
#define USART_TX_Pin GPIO_PIN_2
#define USART_TX_GPIO_Port GPIOA
#define USART_RX_Pin GPIO_PIN_3
#define USART_RX_GPIO_Port GPIOA
#define LD2_Pin GPIO_PIN_5
#define LD2_GPIO_Port GPIOA
#define TMS_Pin GPIO_PIN_13
#define TMS_GPIO_Port GPIOA
#define TCK_Pin GPIO_PIN_14
#define TCK_GPIO_Port GPIOA
#define SWO_Pin GPIO_PIN_3
#define SWO_GPIO_Port GPIOB
/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
