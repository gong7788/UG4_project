(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(clear b1)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(on b8 t0)
		(in-tower b8)
		(on b7 b8)
		(in-tower b7)
		(on b6 b7)
		(in-tower b6)
		(on b5 b6)
		(in-tower b5)
		(on b4 b5)
		(in-tower b4)
		(on b3 b4)
		(in-tower b3)
		(on b2 b3)
		(in-tower b2)
		(on b1 b2)
		(in-tower b1)
		(red b0)
		(blue b0)
		(green b0)
		(red b1)
		(blue b1)
		(green b1)
		(maroon b1)
		(maroon b2)
		(red b3)
		(blue b3)
		(green b3)
		(maroon b3)
		(red b4)
		(blue b4)
		(green b4)
		(maroon b4)
		(green b5)
		(maroon b5)
		(green b6)
		(maroon b6)
		(red b7)
		(blue b7)
		(green b7)
		(maroon b7)
		(blue b8)
		(green b8)
		(maroon b8)
		(red b9)
		(blue b9)
		(green b9)
		(maroon b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?x) (or (not (green ?x)) (exists (?y) (and (maroon ?y) (on ?x ?y))))) (and (not (on b9 t0)) (not (on b9 b8)) (not (on b9 b7)) (not (on b9 b6)) (not (on b9 b5)) (not (on b9 b4)) (not (on b9 b3)) (not (on b9 b2)) (not (on b9 b1)))))
)