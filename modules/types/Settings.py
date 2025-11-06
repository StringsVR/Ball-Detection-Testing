class Settings:
    def __init__(self, gaussBlur, coverage_threshold, circleDp, circleMinDist, circleParam1, circleParam2, circleMinRadius, circleMaxRadius):
        self.gaussBlur = gaussBlur
        self.coverage_threshold = coverage_threshold

        # Circle
        self.circleDp = circleDp
        self.circleMinDist = circleMinDist
        self.circleParam1 = circleParam1
        self.circleParam2 = circleParam2
        self.circleMinRadius = circleMinRadius
        self.circleMaxRadius = circleMaxRadius