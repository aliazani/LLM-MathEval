FROM alpine:latest

# Define environment variables
ARG JMETER_VERSION=5.6.3
ENV JMETER_HOME=/opt/apache-jmeter-${JMETER_VERSION}
ENV JMETER_BIN=${JMETER_HOME}/bin
ENV PATH="${JMETER_BIN}:${PATH}"

# Install required packages including monitoring tools
RUN apk update && apk upgrade && apk add --no-cache \
    openjdk11-jre \
    git \
    curl \
    wget \
    bash \
    unzip \
    python3 \
    py3-pip \
    tzdata \
    ca-certificates \
    nss

# Install JMeter
RUN mkdir -p /opt && \
    curl -L https://dlcdn.apache.org//jmeter/binaries/apache-jmeter-${JMETER_VERSION}.tgz | tar xz -C /opt

# Install JMeter Plugins Manager + Ultimate Thread Group + others
RUN curl -L -o ${JMETER_HOME}/lib/ext/JMeterPlugins-Manager.jar https://jmeter-plugins.org/get/ && \
    curl -L -o ${JMETER_HOME}/lib/cmdrunner-2.3.jar \
      http://search.maven.org/remotecontent?filepath=kg/apc/cmdrunner/2.3/cmdrunner-2.3.jar && \
    java -cp "${JMETER_HOME}/lib/ext/JMeterPlugins-Manager.jar:${JMETER_HOME}/lib/*" \
      org.jmeterplugins.repository.PluginManagerCMDInstaller && \
    ${JMETER_HOME}/bin/PluginsManagerCMD.sh install tilln-sshmon,jpgc-casutg,jpgc-json,jpgc-perfmon

# Configure JMeter properties at build time
RUN { \
    echo "jmeter.save.saveservice.output_format=csv"; \
    echo "jmeter.save.saveservice.timestamp_format=ms"; \
    echo "jmeter.save.saveservice.time=true"; \
    echo "jmeter.save.saveservice.label=true"; \
    echo "jmeter.save.saveservice.thread_name=true"; \
    echo "jmeter.save.saveservice.response_code=true"; \
    echo "jmeter.save.saveservice.successful=true"; \
    echo "jmeter.save.saveservice.bytes=true"; \
    echo "jmeter.save.saveservice.sent_bytes=true"; \
    echo "jmeter.save.saveservice.latency=true"; \
    echo "jmeter.save.saveservice.connect_time=true"; \
    echo "jmeter.save.saveservice.hostname=true"; \
    echo "jmeter.save.saveservice.url=true"; \
    echo "jmeter.save.saveservice.idle_time=true"; \
    echo "sample_variables=prompt,generated_answer"; \
    } >> ${JMETER_HOME}/bin/user.properties

# Create working directory
WORKDIR /app

# Copy your app files
COPY ./app /app

# Copy and set entrypoint
RUN chmod +x /app/entrypoint.sh

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]

