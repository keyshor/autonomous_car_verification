#include "AddedResets.h"

Real sec(Real input){
        Real out = Real(input);
	
	out.cos_assign();

	Real one = Real(1);
	
        return one / out;
}

Real divide(Real input){
  
        return Real(1) / input;
}

Real arctan(double input){

        return Real(atan(input));
}

Real tan(Real input){
        Real denom = Real(input);
	Real num = Real(input);

	num.sin_assign();
	denom.cos_assign();
  
        return num/denom;
}

Real cosine(Real input){

        Real out = Real(input);

	out.cos_assign();
  
        return out;
}

Real sine(Real input){

        Real out = Real(input);

	out.sin_assign();
  
        return out;
}

Real sqrt(Real input){
        Real output = Real(input);

	output.sqrt_assign();
  
        return output;
}

Real sec1stDer(Real input){
        return sec(input) * tan(input);
}

Real sec2ndDer(Real input){
        //return sec(input) * pow(tan(input), 2) + pow(sec(input), 3);
        return sec(input) * tan(input) * tan(input) + sec(input) * sec(input) * sec(input);
}

Real sec3rdDer(Real input){
        //return sec(input) * pow(tan(input), 3) + 5 * pow(sec(input), 3) * tan(input);
        return sec(input) * tan(input) * tan(input) * tan(input) + Real(5) * sec(input) * sec(input) * sec(input) * tan(input);
}

Real sec4thDer(Real input){
        //return sec(input) * pow(tan(input), 4) + 18 * pow(sec(input), 3) * pow(tan(input), 2) + 5 * pow(sec(input), 5);
	return sec(input) * tan(input) * tan(input) * tan(input) * tan(input) +
	  Real(18) * sec(input) * sec(input) * sec(input) * tan(input) * tan(input) +
	  Real(5) * sec(input) * sec(input) * sec(input) * sec(input) * sec(input);
}

Real sec5thDer(Real input){
	return sec(input) * tan(input) * tan(input) * tan(input) * tan(input) * tan(input) +
	  Real(58) * sec(input) * sec(input) * sec(input) * tan(input) * tan(input) * tan(input) +
	  Real(61) * sec(input) * sec(input) * sec(input) * sec(input) * sec(input) * tan(input);
}

Real div1stDer(Real input){
  
        return Real(-1) / (input * input);
}

Real div2ndDer(Real input){
  
        return Real(2) / (input * input * input);
}

Real div3rdDer(Real input){
  
        return Real(-6) / (input * input * input * input);
}

Real div4thDer(Real input){
  
        return Real(24) / (input * input * input * input * input);
}

Real div5thDer(Real input){
  
        return Real(-120) / (input * input * input * input * input * input);
}

Real arctanCoef(int order){

        if (order % 2 == 0) return Real(0);

	if((order + 1) % 4 == 2) return Real(1.0/order);

	else return Real(-1.0/order);
}

Real arctanDer(int order, Real input){

	Real sum = 0;

	bool posSign = true;

	for(int i = 1; i <= order; i++){
	        if (i % 2 != 0){
		        Real nextEl = Real(input);

			nextEl.pow_assign(i);

			if(posSign) sum += nextEl/Real(i);

			else sum -= nextEl/Real(i);

			posSign = !posSign;
		}

	}

	return sum;
}

Real tan1stDer(Real input){
        return Real(1) + tan(input) * tan(input);
}

Real tan2ndDer(Real input){
        return Real(2) * tan(input) + Real(2) * tan(input) * tan(input) * tan(input);
}

Real tan3rdDer(Real input){
        return Real(2) + Real(8) * tan(input) * tan(input) + Real(6) * tan(input) * tan(input) * tan(input) * tan(input);
}

Real tan4thDer(Real input){
        return Real(16) * tan(input) + Real(40) * tan(input) * tan(input) * tan(input) +
	  Real(24) * tan(input) * tan(input) * tan(input) * tan(input) * tan(input);
}

Real cos1stDer(Real input){

        Real out = Real(input);

	out.sin_assign();
  
        return Real(-1.0) * out;
}

Real cos2ndDer(Real input){

        Real out = Real(input);

	out.cos_assign();
  
        return Real(-1.0) * out;
}

Real cos3rdDer(Real input){

        Real out = Real(input);

	out.sin_assign();
  
        return out;
}

Real cos4thDer(Real input){

        Real out = Real(input);

	out.cos_assign();
  
        return out;
}

Real sin1stDer(Real input){

        Real out = Real(input);

	out.cos_assign();
  
        return out;
}

Real sin2ndDer(Real input){

        Real out = Real(input);

	out.sin_assign();
  
        return Real(-1.0) * out;
}

Real sin3rdDer(Real input){

        Real out = Real(input);

	out.cos_assign();
  
        return Real(-1.0) * out;
}

Real sin4thDer(Real input){

        Real out = Real(input);

	out.sin_assign();
  
        return out;
}

Real sqrt1stDer(Real input){

        Real inSqrt = Real(input);

	inSqrt.sqrt_assign();
  
        return Real(0.5) / inSqrt;
}

Real sqrt2ndDer(Real input){
  
        Real inSqrt = Real(input);

	inSqrt.sqrt_assign();
  
        return Real(-0.25) / (inSqrt * Real(input));
}

Real sqrt3rdDer(Real input){
        Real inSqrt = Real(input);

	inSqrt.sqrt_assign();

	Real out = Real( 3.0 / 8.0 ) / (inSqrt * Real(input) * Real(input));
  
        return out;
}

Real sqrt4thDer(Real input){
        Real inSqrt = Real(input);

	inSqrt.sqrt_assign();
  
        return Real(- 15.0 / 16.0 ) / (inSqrt * Real(input) * Real(input) * Real(input));
}

Real getSecDerBound(Interval intC, int order){

        Real derBound;
	if (intC.inf() > -M_PI/2 && intC.sup() < M_PI/2){

	        if (order == 5){
		        derBound = sec5thDer(intC.sup());
			
			if (Real(-1) * sec5thDer(intC.inf()) > derBound)
			        derBound = Real(-1) * sec5thDer(intC.inf());
		}
	  
	        if (order == 4){
		        derBound = sec4thDer(intC.sup());
			
			if (sec4thDer(intC.inf()) > derBound)
			        derBound = sec4thDer(intC.inf());
		}

	        if (order == 3){
		        derBound = sec3rdDer(intC.sup());
			
			if (Real(-1) * sec3rdDer(intC.inf()) > derBound)
			        derBound = Real(-1) * sec3rdDer(intC.inf());
		}	
	}
	else if (intC.inf() > M_PI/2 && intC.sup() <= M_PI){
	        if (order == 5)
		        derBound = sec5thDer(intC.inf());
	  
	        if (order == 4)
		        derBound = Real(-1) * sec4thDer(intC.inf());

		if (order == 3)
		        derBound = sec3rdDer(intC.inf());
	}
	else if (intC.inf() >= -M_PI && intC.sup() < -M_PI/2){
	        if (order == 5)
		        derBound = Real(-1) * sec5thDer(intC.sup());
	  
	        if (order == 4)
		        derBound = Real(-1) * sec4thDer(intC.sup());

		if (order == 3)		  
		        derBound = Real(-1) * sec3rdDer(intC.sup());
	}
	else{
	        printf("Uncertainty too large. Please try decreasing the initial set size.\n");
		exit(-1);
	}

	return derBound;
}

Real getSecRemUpperBound(Interval intC, int order){

        Real derBound;
	if (intC.inf() > -M_PI/2 && intC.sup() < M_PI/2){

	        if (order == 5){
		        derBound = sec5thDer(intC.sup());
		}
	  
	        if (order == 4){
		        derBound = sec4thDer(intC.sup());
			
			if (sec4thDer(intC.inf()) > derBound)
			        derBound = sec4thDer(intC.inf());
		}

	        if (order == 3){
		        derBound = sec3rdDer(intC.sup());
		}	
	}
	else if (intC.inf() > M_PI/2 && intC.sup() <= M_PI){
	        if (order == 5)
		        derBound = sec5thDer(intC.inf());
	  
	        if (order == 4)
		        derBound = sec4thDer(intC.sup());

		if (order == 3)
		        derBound = sec3rdDer(intC.inf());
	}
	else if (intC.inf() >= -M_PI && intC.sup() < -M_PI/2){
	        if (order == 5)
		        derBound = sec5thDer(intC.inf());
	  
	        if (order == 4)
		        derBound = sec4thDer(intC.inf());

		if (order == 3)		  
		        derBound = sec3rdDer(intC.inf());
	}
	else{
	        printf("Uncertainty too large. Please try decreasing the initial set size.\n");
		exit(-1);
	}

	return derBound;
}

Real getSecRemLowerBound(Interval intC, int order){

        Real derBound;
	if (intC.inf() > -M_PI/2 && intC.sup() < M_PI/2){

	        if (order == 5){
		        derBound = sec5thDer(intC.inf());
		}
	  
	        if (order == 4){
		        derBound = sec4thDer(intC.sup());
			
			if (derBound > sec4thDer(intC.inf()))
			        derBound = sec4thDer(intC.inf());
		}

	        if (order == 3){
		        derBound = sec3rdDer(intC.inf());
		}	
	}
	else if (intC.inf() > M_PI/2 && intC.sup() <= M_PI){
	        if (order == 5)
		        derBound = sec5thDer(intC.sup());
	  
	        if (order == 4)
		        derBound = sec4thDer(intC.inf());

		if (order == 3)
		        derBound = sec3rdDer(intC.sup());
	}
	else if (intC.inf() >= -M_PI && intC.sup() < -M_PI/2){
	        if (order == 5)
		        derBound = sec5thDer(intC.sup());
	  
	        if (order == 4)
		        derBound = sec4thDer(intC.sup());

		if (order == 3)		  
		        derBound = sec3rdDer(intC.sup());
	}
	else{
	        printf("Uncertainty too large. Please try decreasing the initial set size.\n");
		exit(-1);
	}

	return derBound;
}

Interval getSecDerRemBound(const Interval inputBounds, const double apprPoint, const int order){
  
        Interval upper = Interval(apprPoint, inputBounds.sup());
	Interval lower = Interval(inputBounds.inf(), apprPoint);
  
        Real Q_u = getSecRemUpperBound(upper, order);
	Real Q_l = getSecRemUpperBound(lower, order);
	Real q_u = getSecRemLowerBound(upper, order);
	Real q_l = getSecRemLowerBound(lower, order);

        Real fact = Real(24);
	if(order == 5)
	  fact = Real(120);
	
	Real maxPosDev = Real(inputBounds.sup() - apprPoint);
	Real maxNegDev = Real(inputBounds.inf()- apprPoint);
	maxPosDev.pow_assign(order);
	maxNegDev.pow_assign(order);

	Real u = (maxPosDev * Q_u) / fact;
	Real l = (maxPosDev * q_u) / fact;

	if((maxNegDev * Q_l) / fact > u) u = (maxNegDev * Q_l) / fact;
	if(l > (maxNegDev * q_l) / fact) l = (maxNegDev * q_l) / fact;

	//these checks are necessary because the remainder is always 0 at apprPoint
	if(Real(0) > u) u = Real(0);
	if(l > Real(0)) l = Real(0);
	
        return Interval(l.getValue_RNDD(), u.getValue_RNDU());
    
}

Real getDivRemUpperBound(const Interval intC, const int order){

        Real bound = Real(0);

        if(order == 3){
	        //3rd derivative is negative and decreasing for negative numbers
	        if (intC.sup() < 0){
		        bound = div3rdDer(Real(intC.inf()));
		}
		//3rd derivative is negative and increasing for positive numbers
		else if (intC.inf() > 0){
		        bound = div3rdDer(Real(intC.sup()));
		}
	}

	else if(order == 4){
	        //4th derivative is negative and decreasing for negative numbers
	        if (intC.sup() < 0){
		        bound = div3rdDer(Real(intC.inf()));
		}
		//4th derivative is positive and decreasing for positive numbers
		else if (intC.inf() > 0){
		        bound = div3rdDer(Real(intC.inf()));
		}
	}

	else if(order == 5){
	        //5th derivative is negative and decreasing for negative numbers
	        if (intC.sup() < 0){
		        bound = div3rdDer(Real(intC.inf()));
		}
		//5th derivative is negative and increasing for positive numbers
		else if (intC.inf() > 0){
		        double maxVal = fabs(div5thDer(Real(intC.inf())).getValue_RNDD());

		        bound = div3rdDer(Real(intC.sup()));
		}
	}

	return bound;
}

Real getDivRemLowerBound(const Interval intC, const int order){

        Real bound = Real(0);

        if(order == 3){
	        //3rd derivative is negative and decreasing for negative numbers
	        if (intC.sup() < 0){
		        bound = div3rdDer(Real(intC.sup()));
		}
		//3rd derivative is negative and increasing for positive numbers
		else if (intC.inf() > 0){
		        bound = div3rdDer(Real(intC.inf()));
		}
	}

	else if(order == 4){
	        //4th derivative is negative and decreasing for negative numbers
	        if (intC.sup() < 0){
		        bound = div3rdDer(Real(intC.sup()));
		}
		//4th derivative is positive and decreasing for positive numbers
		else if (intC.inf() > 0){
		        bound = div3rdDer(Real(intC.sup()));
		}
	}

	else if(order == 5){
	        //5th derivative is negative and decreasing for negative numbers
	        if (intC.sup() < 0){
		        bound = div3rdDer(Real(intC.sup()));
		}
		//5th derivative is negative and increasing for positive numbers
		else if (intC.inf() > 0){
		        bound = div3rdDer(Real(intC.inf()));
		}
	}

	return bound;
}

Interval getDivDerRemBound(const Interval inputBounds, const double apprPoint, const int order){
  
        Interval upper = Interval(apprPoint, inputBounds.sup());
	Interval lower = Interval(inputBounds.inf(), apprPoint);
  
        Real Q_u = getDivRemUpperBound(upper, order);
	Real Q_l = getDivRemUpperBound(lower, order);
	Real q_u = getDivRemLowerBound(upper, order);
	Real q_l = getDivRemLowerBound(lower, order);

        Real fact = Real(24);
	if(order == 5)
	  fact = Real(120);
	
	Real maxPosDev = Real(inputBounds.sup() - apprPoint);
	Real maxNegDev = Real(inputBounds.inf()- apprPoint);
	maxPosDev.pow_assign(order);
	maxNegDev.pow_assign(order);

	Real u = (maxPosDev * Q_u) / fact;
	Real l = (maxPosDev * q_u) / fact;

	if((maxNegDev * Q_l) / fact > u) u = (maxNegDev * Q_l) / fact;
	if(l > (maxNegDev * q_l) / fact) l = (maxNegDev * q_l) / fact;

	//these checks are necessary because the remainder is always 0 at apprPoint
	if(Real(0) > u) u = Real(0);
	if(l > Real(0)) l = Real(0);
	
        return Interval(l.getValue_RNDD(), u.getValue_RNDU());
    
}

Real getDivDerBound(int order, Interval intC){

        Real bound = Real(0);

        if(order == 3){
	        //3rd derivative is negative and decreasing for negative numbers
	        if (intC.sup() < 0){
		        double maxVal = fabs(div3rdDer(Real(intC.sup())).getValue_RNDD());

			bound = Real(maxVal);
		}
		//3rd derivative is negative and increasing for positive numbers
		else if (intC.inf() > 0){
		        double maxVal = fabs(div3rdDer(Real(intC.inf())).getValue_RNDD());

			bound = Real(maxVal);
		}
	}

	else if(order == 4){
	        //4th derivative is negative and decreasing for negative numbers
	        if (intC.sup() < 0){
		        double maxVal = fabs(div4thDer(Real(intC.sup())).getValue_RNDD());

			bound = Real(maxVal);
		}
		//4th derivative is positive and decreasing for positive numbers
		else if (intC.inf() > 0){
		        double maxVal = fabs(div4thDer(Real(intC.inf())).getValue_RNDD());

			bound = Real(maxVal);
		}
	}

	else if(order == 5){
	        //5th derivative is negative and decreasing for negative numbers
	        if (intC.sup() < 0){
		        double maxVal = fabs(div5thDer(Real(intC.sup())).getValue_RNDD());

			bound = Real(maxVal);
		}
		//5th derivative is negative and increasing for positive numbers
		else if (intC.inf() > 0){
		        double maxVal = fabs(div5thDer(Real(intC.inf())).getValue_RNDD());

			bound = Real(maxVal);
		}
	}

	return bound;
}

//NB: this assumes intC \subset [-1, 1]
Real getArcTanDerBound(double dev, int order){

	return Real( (pow(dev, 2 * order + 1)) / (2 * order + 1));
}

Real getTanDerBound(int order, Interval intC){

	if(order == 4){
	        double low = fabs(tan4thDer(Real(intC.inf())).getValue_RNDD());
		double high = fabs(tan4thDer(Real(intC.sup())).getValue_RNDU());

		if (low > high) return Real(low);
		else return Real(high);
		  
	}

	else{ //if order == 3
	        double low = fabs(tan3rdDer(Real(intC.inf())).getValue_RNDD());
		double high = fabs(tan3rdDer(Real(intC.sup())).getValue_RNDU());

		if (low > high) return Real(low);
		else return Real(high);
		  
	}
}

Real getCosDerBound(int order, Interval intC){

        Real bound = Real(1);

        if(order == 3){ //derivative is sin(x)

	        if(intC.sup() - intC.inf() < M_PI){
		        int pio2Mult = ceil((intC.inf() + M_PI/2) / M_PI);

			double rem = pio2Mult * M_PI - intC.inf() - M_PI/2;

			//printf("rem: %f\n", rem);

			if (intC.sup() - intC.inf() < rem){
			        double maxVal = fabs(cos3rdDer(Real(intC.inf())).getValue_RNDD());

				if (fabs(cos3rdDer(Real(intC.sup())).getValue_RNDD()) > maxVal){

				        maxVal = fabs(cos3rdDer(Real(intC.sup())).getValue_RNDD());
					
				}

				bound = Real(maxVal);
			}
		}

	}

	else if(order == 4){ //derivative is cos(x)
	        if(intC.sup() - intC.inf() < M_PI){
		        int pio2Mult = ceil(intC.inf() / M_PI);

			double rem = pio2Mult * M_PI - intC.inf();

			if (intC.sup() - intC.inf() < rem){
			        double maxVal = fabs(cos4thDer(Real(intC.inf())).getValue_RNDD());

				if (fabs(cos4thDer(Real(intC.sup())).getValue_RNDD()) > maxVal){

				        maxVal = fabs(cos4thDer(Real(intC.sup())).getValue_RNDD());
					
				}

				bound = Real(maxVal);
			}
		}
	}

	return bound;

}

Real getSinDerBound(int order, Interval intC){

        Real bound = Real(1);

        if(order == 4){ //derivative is sin(x)

	        if(intC.sup() - intC.inf() < M_PI){
		        int pio2Mult = ceil((intC.inf() + M_PI/2) / M_PI);

			double rem = pio2Mult * M_PI - intC.inf() - M_PI/2;

			//printf("rem: %f\n", rem);

			if (intC.sup() - intC.inf() < rem){
			        double maxVal = fabs(cos3rdDer(Real(intC.inf())).getValue_RNDD());

				if (fabs(cos3rdDer(Real(intC.sup())).getValue_RNDD()) > maxVal){

				        maxVal = fabs(cos3rdDer(Real(intC.sup())).getValue_RNDD());
					
				}

				bound = Real(maxVal);
			}
		}

	}

	else if(order == 3){ //derivative is -cos(x)
	        if(intC.sup() - intC.inf() < M_PI){
		        int pio2Mult = ceil(intC.inf() / M_PI);

			double rem = pio2Mult * M_PI - intC.inf();

			if (intC.sup() - intC.inf() < rem){
			        double maxVal = fabs(cos4thDer(Real(intC.inf())).getValue_RNDD());

				if (fabs(cos4thDer(Real(intC.sup())).getValue_RNDD()) > maxVal){

				        maxVal = fabs(cos4thDer(Real(intC.sup())).getValue_RNDD());
					
				}

				bound = Real(maxVal);
			}
		}
	}

	return bound;

}

Real getSqrtDerBound(int order, Interval intC){

        Real bound = Real(0);

        if(order == 3){ //3rd derivative is positive and decreasing so the max is the absolute value of intC.inf()

	        double maxVal = fabs(sqrt3rdDer(Real(intC.inf())).getValue_RNDD());


		bound = Real(maxVal);
	}

	else if(order == 4){ //4th derivative is (negative and) increasing so the max is the absolute value of intC.inf()
	        double maxVal = fabs(sqrt4thDer(Real(intC.inf())).getValue_RNDD());

		bound = Real(maxVal);

	}

	return bound;

}

void sqrt_reset(TaylorModel &tmReset, const Interval intC, const int varStoreInd, const int varInputInd, const int numVars){

    Polynomial exp;
    Interval rem;

					
    Real midPoint = Real(intC.midpoint());
    
    Real apprPoint = sqrt(intC.midpoint());
    
    //NB: This assumes a 2nd order TS approximation
    Real coef1 = sqrt1stDer(midPoint);
    Real coef2 = sqrt2ndDer(midPoint)/2;
    Real coef3 = sqrt3rdDer(midPoint)/6;
    
    Real derBound = getSqrtDerBound(3, intC);

    Real maxDev = Real(intC.sup()) - midPoint;
    if (midPoint - Real(intC.inf()) > maxDev){
        maxDev = midPoint - Real(intC.inf());
    }

    Real fact = 6;
    maxDev.pow_assign(3);
    
    Real remainder = (derBound * maxDev) / fact;
    
    Interval apprInt = Interval(apprPoint);
    
    Interval deg1Int = Interval(coef1);
    Interval deg2Int = Interval(coef2);
    Interval deg3Int = Interval(coef3);

    std::vector<int> deg1(numVars, 0);
    deg1[varInputInd + 1] = 1;
    std::vector<int> deg2(numVars, 0);
    deg2[varInputInd + 1] = 2;
    std::vector<int> deg3(numVars, 0);
    deg3[varInputInd + 1] = 3;

    Polynomial deg0Poly = Polynomial(Monomial(apprInt, numVars));

    Polynomial deg1Poly = Polynomial(Monomial(Interval(Real(-1) * coef1 * midPoint), numVars)) +
      Polynomial(Monomial(deg1Int, deg1));

    Polynomial deg2Poly = Polynomial(Monomial(Interval(coef2 * midPoint * midPoint), numVars)) -
      Polynomial(Monomial(Interval(Real(2) * coef2 * midPoint), deg1)) +
      Polynomial(Monomial(deg2Int, deg2));
					
    exp = deg0Poly + deg1Poly + deg2Poly;
    remainder.to_sym_int(rem);

    //if uncertainty too large, use a 3rd order approximation
    if (rem.width() > 0.00001){
        fact = 24;
	maxDev = Real(intC.sup()) - midPoint;
	maxDev.pow_assign(4);
	derBound = getSqrtDerBound(4, intC);
						
	remainder = (derBound * maxDev) / fact;
	remainder.to_sym_int(rem);

	Polynomial deg3Poly = Polynomial(Monomial(Interval(Real(-1) * coef3 * midPoint * midPoint * midPoint), numVars)) +
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint * midPoint), deg1)) -
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint), deg2)) +
	  Polynomial(Monomial(deg3Int, deg3));
						
	exp += deg3Poly;
    }					


    if(rem.width() > 0.001){
        printf("Uncertainty too large. Please increase Taylor Model order.\n");
	exit(-1);
    }					

    tmReset.expansion = exp;
    tmReset.remainder = rem;    
}

void sin_reset(TaylorModel &tmReset, const Interval intC, const int varStoreInd, const int varInputInd, const int numVars){

    Polynomial exp;
    Interval rem;

    Real midPoint = Real(intC.midpoint());

    Real apprPoint = sine(intC.midpoint());

    //NB: This assumes a 2nd order TS approximation
    Real coef1 = sin1stDer(midPoint);
    Real coef2 = sin2ndDer(midPoint)/2;
    Real coef3 = sin3rdDer(midPoint)/6;

    Real derBound = getSinDerBound(3, intC);
    
    Real maxDev = Real(intC.sup()) - midPoint;
    if (midPoint - Real(intC.inf()) > maxDev){
        maxDev = midPoint - Real(intC.inf());
    }

    Real fact = 6;
    maxDev.pow_assign(3);
					
    Real remainder = (derBound * maxDev) / fact;
    
    Interval apprInt = Interval(apprPoint);
    
    Interval deg1Int = Interval(coef1);
    Interval deg2Int = Interval(coef2);
    Interval deg3Int = Interval(coef3);
    
    std::vector<int> deg1(numVars, 0);
    deg1[varInputInd + 1] = 1;
    std::vector<int> deg2(numVars, 0);
    deg2[varInputInd + 1] = 2;
    std::vector<int> deg3(numVars, 0);
    deg3[varInputInd + 1] = 3;
					
    Polynomial deg0Poly = Polynomial(Monomial(apprInt, numVars));

    Polynomial deg1Poly = Polynomial(Monomial(Interval(Real(-1) * coef1 * midPoint), numVars)) +
      Polynomial(Monomial(deg1Int, deg1));

    Polynomial deg2Poly = Polynomial(Monomial(Interval(coef2 * midPoint * midPoint), numVars)) -
      Polynomial(Monomial(Interval(Real(2) * coef2 * midPoint), deg1)) +
      Polynomial(Monomial(deg2Int, deg2));
					
    exp = deg0Poly + deg1Poly + deg2Poly;
    remainder.to_sym_int(rem);

    //if uncertainty too large, use a 3rd order approximation
    if (rem.width() > 0.00001){
        fact = 24;
	maxDev = Real(intC.sup()) - midPoint;
	maxDev.pow_assign(4);
	derBound = getSinDerBound(4, intC);
	
	remainder = (derBound * maxDev) / fact;
	remainder.to_sym_int(rem);
	
	Polynomial deg3Poly = Polynomial(Monomial(Interval(Real(-1) * coef3 * midPoint * midPoint * midPoint), numVars)) +
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint * midPoint), deg1)) -
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint), deg2)) +
	  Polynomial(Monomial(deg3Int, deg3));
						
	exp += deg3Poly;
    }					

    if(rem.width() > 0.001){
        printf("Uncertainty too large. Please increase Taylor Model order.\n");
	exit(-1);
    }
    
    tmReset.expansion = exp;
    tmReset.remainder = rem;    
}

void cos_reset(TaylorModel &tmReset, const Interval intC, const int varStoreInd, const int varInputInd, const int numVars){

    Polynomial exp;
    Interval rem;

    Real midPoint = Real(intC.midpoint());

    Real apprPoint = cosine(intC.midpoint());

    //NB: This assumes a 2nd order TS approximation
    Real coef1 = cos1stDer(midPoint);
    Real coef2 = cos2ndDer(midPoint)/2;
    Real coef3 = cos3rdDer(midPoint)/6;
    
    Real derBound = getCosDerBound(3, intC);
					
    Real maxDev = Real(intC.sup()) - midPoint;
    if (midPoint - Real(intC.inf()) > maxDev){
        maxDev = midPoint - Real(intC.inf());
    }

    Real fact = 6;
    maxDev.pow_assign(3);
					
    Real remainder = (derBound * maxDev) / fact;
    
    Interval apprInt = Interval(apprPoint);
    
    Interval deg1Int = Interval(coef1);
    Interval deg2Int = Interval(coef2);
    Interval deg3Int = Interval(coef3);
    
    std::vector<int> deg1(numVars, 0);
    deg1[varInputInd + 1] = 1;
    std::vector<int> deg2(numVars, 0);
    deg2[varInputInd + 1] = 2;
    std::vector<int> deg3(numVars, 0);
    deg3[varInputInd + 1] = 3;

    Polynomial deg0Poly = Polynomial(Monomial(apprInt, numVars));

    Polynomial deg1Poly = Polynomial(Monomial(Interval(Real(-1) * coef1 * midPoint), numVars)) +
      Polynomial(Monomial(deg1Int, deg1));

    Polynomial deg2Poly = Polynomial(Monomial(Interval(coef2 * midPoint * midPoint), numVars)) -
      Polynomial(Monomial(Interval(Real(2) * coef2 * midPoint), deg1)) +
      Polynomial(Monomial(deg2Int, deg2));
					
    exp = deg0Poly + deg1Poly + deg2Poly;
    remainder.to_sym_int(rem);

    //if uncertainty too large, use a 3rd order approximation
    if (rem.width() > 0.00001){
        fact = 24;
	maxDev = Real(intC.sup()) - midPoint;
	maxDev.pow_assign(4);
	derBound = getCosDerBound(4, intC);
						
	remainder = (derBound * maxDev) / fact;
	remainder.to_sym_int(rem);
	
	Polynomial deg3Poly = Polynomial(Monomial(Interval(Real(-1) * coef3 * midPoint * midPoint * midPoint), numVars)) +
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint * midPoint), deg1)) -
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint), deg2)) +
	  Polynomial(Monomial(deg3Int, deg3));
						
	exp += deg3Poly;
    }
    
    if(rem.width() > 0.001){
        printf("Uncertainty too large. Please increase Taylor Model order.\n");
	exit(-1);
    }

    tmReset.expansion = exp;
    tmReset.remainder = rem;    
}

void tan_reset(TaylorModel &tmReset, const Interval intC, const int varStoreInd, const int varInputInd, const int numVars){

    Polynomial exp;
    Interval rem;

    Real midPoint = Real(intC.midpoint());

    Real apprPoint = tan(intC.midpoint());

    //NB: This assumes a 2nd order TS approximation
    Real coef1 = tan1stDer(midPoint);
    Real coef2 = tan2ndDer(midPoint)/2;
    Real coef3 = tan3rdDer(midPoint)/6;
    
    Real derBound = getTanDerBound(3, intC);
					
    Real maxDev = Real(intC.sup()) - midPoint;
    if (midPoint - Real(intC.inf()) > maxDev){
        maxDev = midPoint - Real(intC.inf());
    }

    Real fact = 6;
    maxDev.pow_assign(3);
    
    Real remainder = (derBound * maxDev) / fact;

    Interval apprInt = Interval(apprPoint);
    
    Interval deg1Int = Interval(coef1);
    Interval deg2Int = Interval(coef2);
    Interval deg3Int = Interval(coef3);

    std::vector<int> deg1(numVars, 0);
    deg1[varInputInd + 1] = 1;
    std::vector<int> deg2(numVars, 0);
    deg2[varInputInd + 1] = 2;
    std::vector<int> deg3(numVars, 0);
    deg3[varInputInd + 1] = 3;
    
    Polynomial deg0Poly = Polynomial(Monomial(apprInt, numVars));

    Polynomial deg1Poly = Polynomial(Monomial(Interval(Real(-1) * coef1 * midPoint), numVars)) +
      Polynomial(Monomial(deg1Int, deg1));

    Polynomial deg2Poly = Polynomial(Monomial(Interval(coef2 * midPoint * midPoint), numVars)) -
      Polynomial(Monomial(Interval(Real(2) * coef2 * midPoint), deg1)) +
      Polynomial(Monomial(deg2Int, deg2));
					
    exp = deg0Poly + deg1Poly + deg2Poly;
    remainder.to_sym_int(rem);

    //if uncertainty too large, use a 3rd order approximation
    if (rem.width() > 0.00001){
        fact = 24;
	maxDev = Real(intC.sup()) - midPoint;
	maxDev.pow_assign(4);
	derBound = getTanDerBound(4, intC);
						
	remainder = (derBound * maxDev) / fact;
	remainder.to_sym_int(rem);

	Polynomial deg3Poly = Polynomial(Monomial(Interval(Real(-1) * coef3 * midPoint * midPoint * midPoint), numVars)) +
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint * midPoint), deg1)) -
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint), deg2)) +
	  Polynomial(Monomial(deg3Int, deg3));
						
	exp += deg3Poly;
    }
    
    if(rem.width() > 1){
        printf("Uncertainty too large. Please increase Taylor Model order.\n");
	exit(-1);
    }					

    tmReset.expansion = exp;
    tmReset.remainder = rem;    
    
}

void arc_reset(TaylorModel &tmReset, const Interval intC, const int varStoreInd, const int varInputInd, const int numVars){

    Polynomial exp;
    Interval rem;

    Real midPoint = Real(intC.midpoint());
    
    Real apprPoint = arctan(intC.midpoint());


    //NB: This performs a 3rd order TS approximation
    Real coef1 = arctanCoef(1);
    Real coef2 = arctanCoef(2);
    Real coef3 = arctanCoef(3);

    Real maxDev = Real(intC.sup()) - midPoint;
    if (midPoint - Real(intC.inf()) > maxDev){
        maxDev = midPoint - Real(intC.inf());
    }

    Real remainder = getArcTanDerBound(maxDev.getValue_RNDD(), 3);
    
    Interval apprInt = Interval(apprPoint);
    
    Interval deg1Int = Interval(coef1);
    Interval deg2Int = Interval(coef2);
    Interval deg3Int = Interval(coef3);

    std::vector<int> deg1(numVars, 0);
    deg1[varInputInd + 1] = 1;
    std::vector<int> deg2(numVars, 0);
    deg2[varInputInd + 1] = 2;
    std::vector<int> deg3(numVars, 0);
    deg3[varInputInd + 1] = 3;

    Polynomial deg0Poly = Polynomial(Monomial(apprInt, numVars));

    Polynomial deg1Poly = Polynomial(Monomial(Interval(Real(-1) * coef1 * midPoint), numVars)) +
      Polynomial(Monomial(deg1Int, deg1));

    Polynomial deg2Poly = Polynomial(Monomial(Interval(coef2 * midPoint * midPoint), numVars)) -
      Polynomial(Monomial(Interval(Real(2) * coef2 * midPoint), deg1)) +
      Polynomial(Monomial(deg2Int, deg2));
					
    exp = deg0Poly + deg1Poly + deg2Poly;
    remainder.to_sym_int(rem);

    //if uncertainty too large, use a 3rd order approximation
    if (rem.width() > 0.00001){					       

        remainder = getArcTanDerBound(maxDev.getValue_RNDD(), 4);
	remainder.to_sym_int(rem);

	Polynomial deg3Poly = Polynomial(Monomial(Interval(Real(-1) * coef3 * midPoint * midPoint * midPoint), numVars)) +
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint * midPoint), deg1)) -
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint), deg2)) +
	  Polynomial(Monomial(deg3Int, deg3));
						
	exp += deg3Poly;
    }	

    tmReset.expansion = exp;
    tmReset.remainder = rem;
    
}

void sec_reset(TaylorModel &tmReset, const Interval intC, const int varStoreInd, const int varInputInd, const int numVars, const TaylorModelVec tmvAggregation, const std::vector<Interval> doAggregation){

    Polynomial exp;
    Interval rem;

    //Real midPoint = (Real(intC.sup()) + Real(intC.inf()))/2;
    Real midPoint = Real(intC.midpoint());

    Real apprPoint = sec(midPoint);
					
    //NB: This performs a 4th order TS approximation
    Real coef1 = sec1stDer(midPoint);
    Real coef2 = sec2ndDer(midPoint)/2;
    Real coef3 = sec3rdDer(midPoint)/6;
    Real coef4 = sec4thDer(midPoint)/24;

    //Real derBound = getSecDerBound(intC, 5);

    rem = getSecDerRemBound(intC, intC.midpoint(), 5);
    
    Real maxDev = Real(intC.sup()) - midPoint;
    if (midPoint - Real(intC.inf()) > maxDev){
        maxDev = midPoint - Real(intC.inf());
    }
    
    Real fact = Real(120);
    maxDev.pow_assign(5);
					
    //Real remainder = (derBound * maxDev) / fact;
    
    Interval apprInt = Interval(apprPoint);
    
    Interval deg1Int = Interval(coef1);
    Interval deg2Int = Interval(coef2);
    Interval deg3Int = Interval(coef3);
    Interval deg4Int = Interval(coef4);
    
    std::vector<int> deg1(numVars, 0);
    deg1[varInputInd + 1] = 1;
    std::vector<int> deg2(numVars, 0);
    deg2[varInputInd + 1] = 2;
    std::vector<int> deg3(numVars, 0);
    deg3[varInputInd + 1] = 3;
    std::vector<int> deg4(numVars, 0);
    deg4[varInputInd + 1] = 4;

    /*
      Poly approx. = apprPoint + coef1 * (x - midPoint) 
      + coef2 * (x^2 - 2 * x * midPoint + midPoint^2)
      + coef3 * (x^3 - 3 * x^2 * midpoint + 3 * x * midPoint^2 - midPoint^3)
      + coef4 * (x^4 - 4 * x^3 * midpoint + 6 * x^2 * midPoint^2 - 4 * x * midPoint^3 + midPoint^4)
    */

    Polynomial deg0Poly = Polynomial(Monomial(apprInt, numVars));

    Polynomial deg1Poly = Polynomial(Monomial(Interval(Real(-1) * coef1 * midPoint), numVars)) +
      Polynomial(Monomial(deg1Int, deg1));

    Polynomial deg2Poly = Polynomial(Monomial(Interval(coef2 * midPoint * midPoint), numVars)) -
      Polynomial(Monomial(Interval(Real(2) * coef2 * midPoint), deg1)) +
      Polynomial(Monomial(deg2Int, deg2));

    Polynomial deg3Poly =
      Polynomial(Monomial(Interval(Real(-1) * coef3 * midPoint * midPoint * midPoint), numVars)) +
      Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint * midPoint), deg1)) -
      Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint), deg2)) +
      Polynomial(Monomial(deg3Int, deg3));

    Polynomial deg4Poly =
      Polynomial(Monomial(Interval(coef4 * midPoint * midPoint * midPoint * midPoint), numVars)) +
      Polynomial(Monomial(Interval(Real(-4) * coef4 * midPoint * midPoint * midPoint), deg1)) +
      Polynomial(Monomial(Interval(Real(6) * coef4 * midPoint * midPoint), deg2)) +
      Polynomial(Monomial(Interval(Real(-4) * coef4 * midPoint), deg3)) +
      Polynomial(Monomial(deg4Int, deg4));    
					
    exp = deg0Poly + deg1Poly + deg2Poly + deg3Poly + deg4Poly;
    //remainder.to_sym_int(rem);
    
    tmReset.expansion = exp;
    tmReset.remainder = rem;

}

void div_reset(TaylorModel &tmReset, const Interval intC, const int varStoreInd, const int varDenInd, const int numVars){

    Polynomial exp;
    Interval rem;
				  
    Real midPoint = Real(intC.midpoint());

    Real apprPoint = divide(midPoint);
					
    //NB: This performs a 4th order TS approximation
    Real coef1 = div1stDer(midPoint);
    Real coef2 = div2ndDer(midPoint)/2;
    Real coef3 = div3rdDer(midPoint)/6;
    Real coef4 = div4thDer(midPoint)/24;

    Real maxDev = Real(intC.sup()) - midPoint;
    if (midPoint - Real(intC.inf()) > maxDev){
        maxDev = midPoint - Real(intC.inf());
    }

    Real fact = 120;
    maxDev.pow_assign(5);
    Real derBound = getDivDerBound(5, intC);

    //Real remainder = (derBound * maxDev) / fact;

    rem = getDivDerRemBound(intC, intC.midpoint(), 5);
    
    Interval apprInt = Interval(apprPoint);
    
    Interval deg1Int = Interval(coef1);
    Interval deg2Int = Interval(coef2);
    Interval deg3Int = Interval(coef3);
    Interval deg4Int = Interval(coef4);
    
    std::vector<int> deg1(numVars, 0);
    deg1[varDenInd + 1] = 1;
    std::vector<int> deg2(numVars, 0);
    deg2[varDenInd + 1] = 2;
    std::vector<int> deg3(numVars, 0);
    deg3[varDenInd + 1] = 3;
    std::vector<int> deg4(numVars, 0);
    deg4[varDenInd + 1] = 4;
					
    /*
      Poly approx. = apprPoint + coef1 * (x - midPoint) 
      + coef2 * (x^2 - 2 * x * midPoint + midPoint^2)
      + coef3 * (x^3 - 3 * x^2 * midpoint + 3 * x * midPoint^2 - midPoint^3)
      + coef4 * (x^4 - 4 * x^3 * midpoint + 6 * x^2 * midPoint^2 - 4 * x * midPoint^3 + midPoint^4)
    */
    
    Polynomial deg0Poly = Polynomial(Monomial(apprInt, numVars));
    
    Polynomial deg1Poly = Polynomial(Monomial(Interval(Real(-1) * coef1 * midPoint), numVars)) +
      Polynomial(Monomial(deg1Int, deg1));
    
    Polynomial deg2Poly = Polynomial(Monomial(Interval(coef2 * midPoint * midPoint), numVars)) -
      Polynomial(Monomial(Interval(Real(2) * coef2 * midPoint), deg1)) +
      Polynomial(Monomial(deg2Int, deg2));

    Polynomial deg3Poly =
      Polynomial(Monomial(Interval(Real(-1) * coef3 * midPoint * midPoint * midPoint), numVars)) +
      Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint * midPoint), deg1)) -
      Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint), deg2)) +
      Polynomial(Monomial(deg3Int, deg3));

    Polynomial deg4Poly =
      Polynomial(Monomial(Interval(coef4 * midPoint * midPoint * midPoint * midPoint), numVars)) +
      Polynomial(Monomial(Interval(Real(-4) * coef4 * midPoint * midPoint * midPoint), deg1)) +
      Polynomial(Monomial(Interval(Real(6) * coef4 * midPoint * midPoint), deg2)) +
      Polynomial(Monomial(Interval(Real(-4) * coef4 * midPoint), deg3)) +
      Polynomial(Monomial(deg4Int, deg4));    
					
    exp = deg0Poly + deg1Poly + deg2Poly + deg3Poly + deg4Poly;
    //    remainder.to_sym_int(rem);

    tmReset.expansion = exp;
    tmReset.remainder = rem;
    
}
