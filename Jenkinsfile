pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh '''
                    cd build
                    cmake ..
                    make
                    ctest
                '''
            }
        }
    }
}
