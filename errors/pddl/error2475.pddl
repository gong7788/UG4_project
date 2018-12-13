(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(clear b4)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b5 b9)
		(in-tower b5)
		(on b3 b5)
		(in-tower b3)
		(on b8 b3)
		(in-tower b8)
		(on b7 b8)
		(in-tower b7)
		(on b6 b7)
		(in-tower b6)
		(on b4 b6)
		(in-tower b4)
		(green b0)
		(maroon b0)
		(blue b1)
		(maroon b1)
		(red b2)
		(blue b2)
		(green b2)
		(maroon b2)
		(blue b3)
		(green b3)
		(maroon b3)
		(blue b4)
		(green b4)
		(maroon b4)
		(red b5)
		(blue b5)
		(green b5)
		(maroon b5)
		(green b6)
		(blue b7)
		(maroon b7)
		(green b8)
		(maroon b8)
		(blue b9)
		(green b9)
		(maroon b9)
		(green b1)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?x) (or (not (green ?x)) (exists (?y) (and (maroon ?y) (on ?x ?y))))) (and (not (on b8 b9)) (not (on b7 b9)) (not (on b6 b9)) (not (on b8 b5)) (not (on b7 b5)) (not (on b6 b5)) (not (on b4 b5)) (not (on b2 b4)))))
)