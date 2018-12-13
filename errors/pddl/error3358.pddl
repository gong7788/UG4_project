(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b2)
		(clear b2)
		(clear b5)
		(on-table b6)
		(clear b6)
		(on-table b7)
		(clear b7)
		(on-table b8)
		(clear b8)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(on b0 t0)
		(in-tower b0)
		(on b1 b0)
		(in-tower b1)
		(on b3 b1)
		(in-tower b3)
		(on b4 b3)
		(in-tower b4)
		(on b5 b4)
		(in-tower b5)
		(maroon b0)
		(maroon b1)
		(red b2)
		(blue b2)
		(green b2)
		(maroon b2)
		(green b3)
		(maroon b3)
		(maroon b4)
		(blue b5)
		(maroon b5)
		(blue b6)
		(green b6)
		(maroon b6)
		(blue b7)
		(maroon b7)
		(blue b8)
		(green b8)
		(maroon b8)
		(green b9)
		(maroon b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?x) (or (not (green ?x)) (exists (?y) (and (maroon ?y) (on ?x ?y))))) (and (not (on b2 b1)) (not (on b2 b5)))))
)