# 컨테이너화된 배포

컨테이너화된 환경에서 Nginx를 배포하는 것은 현대적인 애플리케이션 개발과 배포의 핵심입니다. 이 장에서는 Docker와 Kubernetes 환경에서 Nginx를 효과적으로 배포하고 운영하는 방법을 다룹니다.

## Docker를 이용한 Nginx 배포

### 문제 시나리오: 개발 환경과 프로덕션 환경의 일관성
**상황**: 개발 환경에서는 잘 작동하지만 프로덕션 환경에서는 Nginx 설정이 다르게 동작합니다.

**원인 분석**: 환경 간의 차이로 인한 설정 불일치 문제입니다.

**해결 과정**:

#### 1. 기본 Dockerfile 작성

```dockerfile
# Dockerfile
FROM nginx:alpine

# 설정 파일 복사
COPY nginx.conf /etc/nginx/nginx.conf
COPY conf.d/ /etc/nginx/conf.d/

# 정적 콘텐츠 복사
COPY html/ /usr/share/nginx/html/

# SSL 인증서 복사 (필요시)
COPY ssl/ /etc/nginx/ssl/

# 로그 디렉토리 생성
RUN mkdir -p /var/log/nginx/

# 포트暴露
EXPOSE 80 443

# Nginx 시작
CMD ["nginx", "-g", "daemon off;"]
```

#### 2. 멀티스테이지 Dockerfile

```dockerfile
# Dockerfile
# 빌드 스테이지
FROM node:16-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

# 프로덕션 스테이지
FROM nginx:alpine

# 빌드 결과물 복사
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx 설정 복사
COPY nginx.conf /etc/nginx/nginx.conf
COPY conf.d/ /etc/nginx/conf.d/

# 최적화된 Nginx 설정
RUN sed -i 's/user nginx;//' /etc/nginx/nginx.conf && \
    sed -i 's/worker_processes auto;/worker_processes 1;/' /etc/nginx/nginx.conf

# 비루트 사용자로 실행 (보안 강화)
RUN addgroup -g 1001 -S nginx && \
    adduser -S -D -H -u 1001 -h /var/cache/nginx -s /sbin/nolog -G nginx -g nginx nginx

USER nginx

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
```

#### 3. Docker Compose 설정

```yaml
# docker-compose.yml
version: '3.8'

services:
  nginx:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./html:/usr/share/nginx/html:ro
      - ./conf.d:/etc/nginx/conf.d:ro
      - ./ssl:/etc/nginx/ssl:ro
      - ./logs:/var/log/nginx
    depends_on:
      - app1
      - app2
    networks:
      - frontend
      - backend
    restart: unless-stopped

  app1:
    image: node:16-alpine
    working_dir: /app
    volumes:
      - ./app1:/app
    command: npm start
    networks:
      - backend
    restart: unless-stopped

  app2:
    image: node:16-alpine
    working_dir: /app
    volumes:
      - ./app2:/app
    command: npm start
    networks:
      - backend
    restart: unless-stopped

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge

volumes:
  logs:
```

## Kubernetes에서 Nginx 배포

### 문제 시나리오: 마이크로서비스 아키텍처에서의 확장성
**상황**: 트래픽에 따라 자동으로 확장되는 로드 밸런서가 필요합니다.

**해결 과정**:

#### 1. 기본 Nginx Deployment

```yaml
# nginx-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.21-alpine
        ports:
        - containerPort: 80
        volumeMounts:
        - name: nginx-config
          mountPath: /etc/nginx/conf.d
          readOnly: true
        - name: nginx-html
          mountPath: /usr/share/nginx/html
          readOnly: true
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: nginx-config
        configMap:
          name: nginx-config
      - name: nginx-html
        configMap:
          name: nginx-html
```

#### 2. Nginx Service 설정

```yaml
# nginx-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  selector:
    app: nginx
  ports:
    - name: http
      protocol: TCP
      port: 80
      targetPort: 80
  type: ClusterIP
```

#### 3. Ingress 설정

```yaml
# nginx-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nginx-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - example.com
    - www.example.com
    secretName: example-tls
  rules:
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nginx-service
            port:
              number: 80
  - host: www.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nginx-service
            port:
              number: 80
```

#### 4. ConfigMap 설정

```yaml
# nginx-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-config
data:
  default.conf: |
    upstream backend {
        server app1-service:3000;
        server app2-service:3000;
    }
    
    server {
        listen 80;
        server_name _;
        
        location / {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
  
  nginx.conf: |
    user nginx;
    worker_processes auto;
    error_log /var/log/nginx/error.log warn;
    pid /var/run/nginx.pid;
    
    events {
        worker_connections 1024;
    }
    
    http {
        include /etc/nginx/mime.types;
        default_type application/octet-stream;
        
        log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                        '$status $body_bytes_sent "$http_referer" '
                        '"$http_user_agent" "$http_x_forwarded_for"';
        
        access_log /var/log/nginx/access.log main;
        
        sendfile on;
        tcp_nopush on;
        tcp_nodelay on;
        keepalive_timeout 65;
        
        include /etc/nginx/conf.d/*.conf;
    }
```

## Helm을 이용한 Nginx 배포

### 문제 시나리오: 여러 환경에 일관된 Nginx 배포
**상황**: 개발, 스테이징, 프로덕션 환경에 각각 다른 설정으로 Nginx를 배포해야 합니다.

**해결 과정**:

#### 1. Helm Chart 구조

```
nginx-chart/
├── Chart.yaml
├── values.yaml
├── values-dev.yaml
├── values-staging.yaml
├── values-prod.yaml
└── templates/
    ├── deployment.yaml
    ├── service.yaml
    ├── ingress.yaml
    ├── configmap.yaml
    └── secret.yaml
```

#### 2. Chart.yaml

```yaml
# Chart.yaml
apiVersion: v2
name: nginx
description: Nginx Helm Chart
type: application
version: 0.1.0
appVersion: "1.21.0"
```

#### 3. values.yaml

```yaml
# values.yaml
replicaCount: 3

image:
  repository: nginx
  pullPolicy: IfNotPresent
  tag: "1.21-alpine"

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: example-tls
      hosts:
        - example.com

resources:
  limits:
    cpu: 500m
    memory: 128Mi
  requests:
    cpu: 250m
    memory: 64Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80

config:
  upstreamServers:
    - name: app1
      host: app1-service
      port: 3000
    - name: app2
      host: app2-service
      port: 3000
```

#### 4. Deployment 템플릿

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "nginx.fullname" . }}
  labels:
    {{- include "nginx.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "nginx.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "nginx.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 80
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /
              port: http
          readinessProbe:
            httpGet:
              path: /
              port: http
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          volumeMounts:
            - name: nginx-config
              mountPath: /etc/nginx/conf.d
              readOnly: true
      volumes:
        - name: nginx-config
          configMap:
            name: {{ include "nginx.fullname" . }}
```

## 컨테이너 최적화

### 문제 시나리오: 컨테이너 이미지 크기와 시작 시간 최적화
**상황**: 컨테이너 이미지가 너무 크고 시작 시간이 오래 걸립니다.

**해결 과정**:

#### 1. 최적화된 Dockerfile

```dockerfile
# 최적화된 Dockerfile
FROM nginx:1.21-alpine AS base

# 불필요한 패키지 제거
RUN apk del --no-cache nginx-doc && \
    addgroup -g 1001 -S nginx && \
    adduser -S -D -H -u 1001 -h /var/cache/nginx -s /sbin/nolog -G nginx -g nginx nginx

# 최적화된 Nginx 설정
COPY nginx.conf /etc/nginx/nginx.conf
RUN sed -i 's/worker_processes auto;/worker_processes 1;/' /etc/nginx/nginx.conf && \
    sed -i 's/worker_connections 1024;/worker_connections 512;/' /etc/nginx/nginx.conf && \
    mkdir -p /var/cache/nginx /var/log/nginx /var/run && \
    chown -R nginx:nginx /var/cache/nginx /var/log/nginx /var/run

USER nginx

EXPOSE 8080

# 헬스 체크 스크립트
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
```

#### 2. 다단계 빌드 정적 사이트

```dockerfile
# 정적 사이트 빌드
FROM node:16-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

COPY . .
RUN npm run build

# 경량 운영 환경
FROM nginx:1.21-alpine

# 빌드 결과물만 복사
COPY --from=builder /app/dist /usr/share/nginx/html

# 최소한의 설정
COPY nginx.conf /etc/nginx/nginx.conf

# 불필요한 파일 제거
RUN rm -rf /var/cache/apt/* && \
    rm -rf /etc/nginx/conf.d/* && \
    rm -rf /usr/share/nginx/html/* && \
    addgroup -g 1001 -S nginx && \
    adduser -S -D -H -u 1001 -h /var/cache/nginx -s /sbin/nolog -G nginx -g nginx nginx

USER nginx

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
```

## 모니터링 및 로깅

### 문제 시나리오: 컨테이너화된 환경에서의 모니터링
**상황**: 컨테이너가 동적으로 생성되고 종료되어 모니터링이 어렵습니다.

**해결 과정**:

#### 1. Prometheus 모니터링 설정

```yaml
# nginx-prometheus.yaml
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: nginx-metrics
  labels:
    app: nginx
spec:
  selector:
    matchLabels:
      app: nginx
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

#### 2. Nginx Prometheus 모듈 설정

```dockerfile
# Dockerfile with Prometheus
FROM nginx:1.21-alpine

# nginx-prometheus-exporter 설치
RUN apk add --no-cache curl

# Prometheus 모듈 설치
COPY nginx-prometheus.conf /etc/nginx/conf.d/nginx-prometheus.conf

# Prometheus exporter 복사
COPY nginx-prometheus-exporter /usr/local/bin/

# 실행 스크립트
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
```

#### 3. 로깅 설정

```yaml
# fluentd-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*nginx*.log
      pos_file /var/log/fluentd-nginx.log.pos
      tag kubernetes.*
      format json
      time_format %Y-%m-%dT%H:%M:%S.%NZ
    </source>
    
    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      index_name nginx-logs
      type_name _doc
    </match>
```

## 모범 사례 요약

1. **이미지 최적화**: 멀티스테이지 빌드로 불필요한 파일 제거
2. **비루트 실행**: 보안 강화를 위해 비루트 사용자로 실행
3. **헬스 체크**: 적절한 헬스 체크와 레디니스 프로브 설정
4. **자원 제한**: CPU와 메모리 제한으로 안정적인 실행 보장
5. **설정 외부화**: ConfigMap과 Secret을 사용한 설정 관리
6. **로깅**: 중앙화된 로깅 시스템 연동
7. **모니터링**: Prometheus를 이용한 메트릭 수집

## 다음 단계

이제 컨테이너화된 배포를 마쳤습니다. 다음 장에서는 문제 해결 및 디버깅에 대해 알아보겠습니다.